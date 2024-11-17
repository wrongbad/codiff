#pragma once

#include <concepts>

namespace gaii {


template<class T = float>
struct var
{
    T value = 0;
    T grad = 0;

    // var() = default;
    // var(auto && v)
    // : value(std::forward<decltyp(v)>(v))
    // {}

    T & get_value() { return value; }
    T & get_grad() { return grad; }
    T const& get_value() const { return value; }
    T const& get_grad() const { return grad; }
    void backward(auto && grad) { this->grad += grad; }
};

template<class T> var(T) -> var<T>;


// template<class T>
// struct var_traits;

// template<class T>
// struct var_traits<var<T>>
// {
//     using value_type = T;
//     static const bool diffable = true;
// };

// // template<class T>
// // using var_value_type = typename var_traits<std::remove_cvref_t<T>>::value_type;


// template<class T>
// requires std::integral<T> || std::floating_point<T>
// struct var_traits<T>
// {
//     using value_type = T;
//     static const bool diffable = true;
// };

// template<class T>
// T & value(var<T> & v) { return v.value; }

// template<class T>
// T const& value(var<T> const& v) { return v.value; }


template<class T>
concept diffable = requires(T a)
{
    { a.backward(a.get_value()) } -> std::same_as<void>;
};




template<diffable T>
auto & value(T && v)
{
    return v.get_value();
}

template<diffable T>
auto & grad(T && v)
{
    return v.get_grad();
}

template<diffable T>
void backward(T && v, auto && grad)
{
    v.backward(grad);
}




template<class T>
requires std::integral<T> || std::floating_point<T>
T const& value(T const& v)
{ 
    return v;
}

template<class T>
requires std::integral<T> || std::floating_point<T>
void backward(T const&, auto &&)
{   // no-op
} 



} // namespace gaii