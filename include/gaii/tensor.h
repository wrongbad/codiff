#pragma once

#include <concepts>
#include <algorithm>

namespace gaii {


template<class T, int... N>
struct tensor;


template<class T>
struct is_tensor_helper { static constexpr bool value = false; };

template<class T, int... N>
struct is_tensor_helper<tensor<T, N...>> { static constexpr bool value = true; };

template<class T>
concept tensor_ref = is_tensor_helper<std::remove_cvref_t<T>>::value;

template<class T>
concept scalar_ref = std::integral<std::remove_cvref_t<T>> || std::floating_point<std::remove_cvref_t<T>>;

template<class T>
concept tensor_arg = tensor_ref<T> || scalar_ref<T>;


template<tensor_ref T> 
T && as_tensor(T && a) { return std::forward<T>(a); }

template<scalar_ref T>
tensor<std::remove_cvref_t<T>> as_tensor(T a) { return {a}; }

template<tensor_ref Ta, tensor_arg Tb>
Ta & assign(Ta & A, Tb && B);


template<class T>
struct tensor<T>
{
    using element_type = T;

    T m_data;

    static constexpr int size() { return 1; }
    static constexpr int ndim() { return 0; }

    tensor() = default;
    tensor(tensor &&) = default;
    tensor(tensor const&) = default;
    
    template<scalar_ref Tb>
    tensor(Tb b) : m_data(b) {}

    tensor & operator=(tensor &&) = default;
    tensor & operator=(tensor const&) = default;

    template<scalar_ref Tb>
    tensor & operator=(Tb && b) { m_data=b; return *this; }

    T * raw() { return &m_data; }
    T const* raw() const { return &m_data; }

    T & item() { return m_data; }
    T const& item() const { return m_data; }

    operator T&() { return m_data; }
    operator T const&() const { return m_data; }

    template<class F>
    tensor & apply(F && f) { f(m_data); return *this; }
};


template<class T, int N0, int... Ns>
struct tensor<T, N0, Ns...>
{
    using element_type = T;
    using subtensor = tensor<T, Ns...>;

    subtensor m_data[N0];

    static constexpr int size() { return N0 * subtensor::size(); }
    static constexpr int size(int i) { return (int[]){N0, Ns...}[i]; }
    static constexpr int ndim() { return 1 + sizeof...(Ns); }
    
    tensor() = default;

    template<tensor_arg Tb>
    tensor(Tb && b) { assign(*this, std::forward<Tb>(b)); }

    template<class Tb>
    tensor(std::initializer_list<Tb> data) { *this = data; }
    
    template<tensor_arg Tb>
    tensor & operator=(Tb && b) { return assign(*this, std::forward<Tb>(b)); }

    template<class Tb>
    tensor & operator=(std::initializer_list<Tb> data)
    {
        auto it = data.begin();
        for(int i=0 ; i<N0 && it!=data.end() ; i++)
        {
            m_data[i] = *(it++);
        }
        return *this;
    }

    T * raw() { return m_data[0].raw(); }
    T const* raw() const { return m_data[0].raw(); }

    auto & operator[](int i0) { return m_data[i0]; }
    auto & operator[](int i0) const { return m_data[i0]; }

    auto & operator()(int i0) { return m_data[i0]; }
    auto & operator()(int i0) const { return m_data[i0]; }

    template<class... Ints>
    auto & operator()(int i0, Ints... ix) { return m_data[i0](ix...); }
    template<class... Ints>
    auto & operator()(int i0, Ints... ix) const { return m_data[i0](ix...); }

    template<class F>
    tensor & apply(F && f)
    {
        for(int i=0 ; i<N0 ; i++) { m_data[i].apply(f); }
        return *this;
    }
};

template<tensor_ref T>
using element_type = typename T::element_type;

template<class ostream, class T>
ostream & operator<<(ostream & os, tensor<T> const& a)
{
    return os << a.item();
}

template<class ostream, class T, int N0, int... Ns>
ostream & operator<<(ostream & os, tensor<T, N0, Ns...> const& a)
{
    os << "[";
    for(int i=0 ; i<N0 ; i++)
    { 
        if(i) { os << ","; } 
        os << a[i];
    }
    return os << "]";
}



template<int N0, class Tensor>
struct stack_helper;

template<int N0, class T, int... Ns>
struct stack_helper<N0, tensor<T, Ns...>>
{
    using type = tensor<T, N0, Ns...>;
};

template<int N0, class Tensor>
using stack_t = typename stack_helper<N0, std::remove_cvref_t<Tensor>>::type;

template<int N, class Gen>
auto stack(Gen && gen)
{
    if constexpr (std::is_same_v<decltype(gen(0)), void>)
    {
        for(int i=0 ; i<N ; i++) { gen(i); }
    }
    else
    {
        stack_t<N, decltype(as_tensor(gen(0)))> out;
        for(int i=0 ; i<N ; i++) { out[i] = gen(i); }
        return out;
    }
}


template<int OpDim, class ARef, class Op>
auto broadcast(ARef && a, Op op)
{
    using ATensor = std::remove_reference_t<ARef>;
    static_assert(ATensor::ndim() >= OpDim,
        "tensor ndim < broadcast op ndim");

    if constexpr ( ATensor::ndim() > OpDim )
    {
        return stack<ATensor::size(0)>([&] (int i) { 
            return broadcast<OpDim>(a[i], op); 
        });
    }
    else
    {
        return op(a);
    }
}


template<int OpDim, tensor_ref ARef, tensor_ref BRef, class Op>
auto broadcast(ARef && a, BRef && b, Op && op)
{
    // numpy broadcasting rules
    using ATensor = std::remove_reference_t<ARef>;
    using BTensor = std::remove_reference_t<BRef>;
    static_assert(ATensor::ndim() >= OpDim && BTensor::ndim() >= OpDim,
        "tensor ndim < broadcast op ndim");

    if constexpr ( ATensor::ndim() > BTensor::ndim() )
    {
        // iterate leading dims in A, broadcast B
        return stack<ATensor::size(0)>([&] (int i) {
            return broadcast<OpDim>(a[i], b, op);
        });
    }
    else if constexpr ( ATensor::ndim() < BTensor::ndim() )
    {
        // iterate leading dims in B, broadcast A
        return stack<BTensor::size(0)>([&] (int i) {
            return broadcast<OpDim>(a, b[i], op);
        });
    }
    else if constexpr ( ATensor::ndim() > OpDim )
    {
        // iterate common dim in A & B, broadcast if needed
        constexpr int A0 = ATensor::size(0);
        constexpr int B0 = BTensor::size(0);
        constexpr int N0 = A0>B0 ? A0 : B0;
        static_assert(A0==B0 || A0==1 || B0==1, "broadcast mismatch dim");

        return stack<N0>([&] (int i) {
            return broadcast<OpDim>(a[A0>1 ? i : 0], b[B0>1 ? i : 0], op);
        });
    }
    else
    {
        // apply the kernel now that we have target chunk size
        return op(a, b);
    }
}


template<tensor_ref Ta, tensor_arg Tb>
Ta & assign(Ta & A, Tb && B)
{
    broadcast<0>(A, as_tensor(B),
        [] (auto & a, auto & b) { a.item() = b.item(); });
    return A;
}

template<tensor_ref Ta>
Ta & clamp_inplace(Ta & A, element_type<Ta> minv, element_type<Ta> maxv)
{
    return A.apply([&] (auto & a) { a = std::max(minv, std::min(a, maxv)); });
}

template<tensor_ref Ta>
auto operator-(Ta const& A)
{
    return broadcast<0>(A, [] (auto & a) { return -a.item(); });
}


template<class Ta, class Tb> 
requires tensor_ref<Ta> || tensor_ref<Tb>
auto operator+(Ta const& A, Tb const& B)
{
    return broadcast<0>(as_tensor(A), as_tensor(B),
        [] (auto & a, auto & b) { return a.item() + b.item(); });
}

template<class Ta, class Tb> 
requires tensor_ref<Ta> || tensor_ref<Tb>
auto operator-(Ta const& A, Tb const& B)
{
    return broadcast<0>(as_tensor(A), as_tensor(B),
        [] (auto & a, auto & b) { return a.item() - b.item(); });
}

template<class Ta, class Tb> 
requires tensor_ref<Ta> || tensor_ref<Tb>
auto operator*(Ta const& A, Tb const& B)
{
    return broadcast<0>(as_tensor(A), as_tensor(B),
        [] (auto & a, auto & b) { return a.item() * b.item(); });
}

template<class Ta, class Tb> 
requires tensor_ref<Ta> || tensor_ref<Tb>
auto operator/(Ta const& A, Tb const& B)
{
    return broadcast<0>(as_tensor(A), as_tensor(B),
        [] (auto & a, auto & b) { return a.item() / b.item(); });
}


template<tensor_ref Ta, tensor_arg Tb>
Ta & operator+=(Ta & A, Tb const& B)
{
    broadcast<0>(A, as_tensor(B),
        [] (auto & a, auto & b) { a.item() += b.item(); });
    return A;
}

template<tensor_ref Ta, tensor_arg Tb>
Ta & operator-=(Ta & A, Tb const& B)
{
    broadcast<0>(A, as_tensor(B),
        [] (auto & a, auto & b) { a.item() -= b.item(); });
    return A;
}

template<tensor_ref Ta, tensor_arg Tb>
Ta & operator*=(Ta & A, Tb const& B)
{
    broadcast<0>(A, as_tensor(B),
        [] (auto & a, auto & b) { a.item() *= b.item(); });
    return A;
}

template<tensor_ref Ta, tensor_arg Tb>
Ta & operator/=(Ta & A, Tb const& B)
{
    broadcast<0>(A, as_tensor(B),
        [] (auto & a, auto & b) { a.item() /= b.item(); });
    return A;
}


template<class A, class B>
using bin_op_t = decltype(std::declval<A>() * std::declval<B>());


template<int J, int K, int dA=1, class Ta, class Tb>
auto vec_mat_mul(Ta const* a, tensor<Tb, J, K> const& b)
{
    tensor<bin_op_t<Ta, Tb>, K> out;
    constexpr int TILE = 16;
    int k0;
    // break K into tiles for better SIMD utilization
    for(k0=0 ; k0+TILE<=K ; k0+=TILE)
    {
        for(int k=k0 ; k<k0+TILE ; k++) { out(k) = 0; }
        for(int j=0 ; j<J ; j++)
            for(int k=k0 ; k<k0+TILE ; k++)
            {
                out(k) += a[dA * j] * b(j, k).item();
            }
    }
    for(int k=k0 ; k<K ; k++) { out(k) = 0; }
    for(int j=0 ; j<J ; j++)
        for(int k=k0 ; k<K ; k++)
        {
            out(k) += a[dA * j] * b(j, k).item();
        }
    return out;
}

template<bool transA, bool transB>
struct mat_mul_kernel;

template<>
struct mat_mul_kernel<false, false>
{
    // TODO specialize K==1
    template<class Ta, class Tb, int I, int J, int K>
    auto operator()(tensor<Ta, I, J> const& a, tensor<Tb, J, K> const& b)
    {
        tensor<bin_op_t<Ta, Tb>, I, K> out;
        for(int i=0 ; i<I ; i++)
        {
            out[i] = vec_mat_mul(a[i].raw(), b);
        }
        return out;
    }
};

template<>
struct mat_mul_kernel<true, false>
{
    template<class Ta, class Tb, int I, int J, int K>
    auto operator()(tensor<Ta, J, I> const& a, tensor<Tb, J, K> const& b)
    {
        tensor<bin_op_t<Ta, Tb>, I, K> out;
        for(int i=0 ; i<I ; i++)
        {
            // out[i,0:K] = a[0:J, i] @ b[0:J, 0:K]
            out[i] = vec_mat_mul<J, K, I>(a.raw() + i, b);
        }
        return out;
    }
};

template<>
struct mat_mul_kernel<false, true>
{
    template<class Ta, class Tb, int I, int J, int K>
    auto operator()(tensor<Ta, I, J> const& a, tensor<Tb, K, J> const& b)
    {
        tensor<bin_op_t<Ta, Tb>, I, K> out;
        // TODO optimize
        for(int i=0 ; i<I ; i++)
        {
            for(int k=0 ; k<K ; k++)
            {
                out(i,k) = 0;
                for(int j=0 ; j<J ; j++)
                {
                    out(i,k) += a(i,j) * b(k,j);
                }
            }
        }
        return out;
    }
};



template<bool transA, bool transB, tensor_ref Ta, tensor_ref Tb>
auto mat_mul(Ta const& a, Tb const& b)
{
    return broadcast<2>(a, b, mat_mul_kernel<transA, transB>{});
}

template<tensor_ref Ta, tensor_ref Tb>
auto operator%(Ta const& a, Tb const& b)
{
    return mat_mul<false, false>(a, b);
}




template<tensor_ref Ta>
auto sum(Ta const& a)
{
    tensor<element_type<Ta>> out = 0;
    broadcast<0>(out, a, [] (auto & a, auto & b) { a.item() += b; });
    return out;
}

template<tensor_ref Ta>
auto max(Ta const& a)
{
    static_assert(Ta::size() > 0, "max requires > 0 elements");
    tensor<element_type<Ta>> out = *a.raw();
    broadcast<0>(a, [&] (auto & a) { out = std::max(out.item(), a.item()); });
    return out;
}


#define APPROX_MATH

#ifdef APPROX_MATH

float fast_exp(float a)
{
    // https://github.com/ekmett/approximate/blob/master/cbits/fast.c
    union { float f; int32_t i; } p, n;
    p.i = 1056478197 + 6051102 * a; // exp(a/2)
    n.i = 1056478197 - 6051102 * a; // exp(-a/2)
    return p.f / n.f;
}
float fast_log(float a)
{
    // https://github.com/ekmett/approximate/blob/master/cbits/fast.c
    union { float f; int32_t i; } u = { a };
    return (u.i - 1064866805) * (1.0f / 12102203);
}
float fast_sigmoid(float a)
{
    // https://github.com/ekmett/approximate/blob/master/cbits/fast.c
    union { float f; int32_t i; } p, n;
    p.i = 1056478197 + 6051102 * a; // exp(a/2)
    n.i = 1056478197 - 6051102 * a; // exp(-a/2)
    return p.f / (p.f + n.f);
}
float fast_tanh(float a)
{
    // https://github.com/ekmett/approximate/blob/master/cbits/fast.c
    union { float f; int32_t i; } p, n;
    p.i = 1064866805 + 12102203 * a; // exp(a)
    n.i = 1064866805 - 12102203 * a; // exp(-a)
    return (p.f - n.f) / (p.f + n.f);
}

template<tensor_ref Ta>
Ta exp(Ta const& A)
{
    return broadcast<0>(A, [] (auto & a) { return fast_exp(a); });
}

template<tensor_ref Ta>
Ta log(Ta const& A)
{
    return broadcast<0>(A, [] (auto & a) { return fast_log(a); });
}

template<tensor_ref Ta>
Ta sigmoid(Ta const& A)
{
    return broadcast<0>(A, [] (auto & a) { return fast_sigmoid(a); });
}

template<tensor_ref Ta>
Ta tanh(Ta const& A)
{
    return broadcast<0>(A, [] (auto & a) { return fast_tanh(a); });
}

#else // not APPROX_MATH

template<tensor_ref Ta>
Ta exp(Ta const& A)
{
    return broadcast<0>(A, std::exp);
}

template<tensor_ref Ta>
Ta log(Ta const& A)
{
    return broadcast<0>(A, std::log);
}

template<tensor_ref Ta>
Ta tanh(Ta const& A)
{
    return broadcast<0>(A, std::tanh);
}

template<tensor_ref Ta>
Ta sigmoid(Ta const& A)
{
    return broadcast<0>(A, [] (auto & a) {
        return 1 / (1 + std::exp(-a.item()));
    });
}

#endif // APPROX_MATH

template<tensor_ref Ta>
Ta sqrt(Ta const& A)
{
    return broadcast<0>(A, [] (auto & a) { return std::sqrt(a.item()); });
}


// TODO template Dim
// template<class Ta, int... N, int Nlast>
template<tensor_ref Ta>
auto logsumexp(Ta && A)
{
    return broadcast<1>(A, [] (auto & x) {
        auto mx = max(x);
        return log(sum(exp(x - mx))) + mx;
    });
}

// TODO template Dim
template<class Ta, int... N>
tensor<Ta, N...> log_softmax(tensor<Ta, N...> const& A)
{
    return broadcast<1>(A, [] (auto & x) {
        return x - logsumexp(x);
    });
}


} // gaii