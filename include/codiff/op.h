#pragma once

#include "codiff/var.h"

#include <exception>

#if __has_include(<coroutine>)
#include <coroutine>
namespace codiff {
    using std::coroutine_handle;
    using std::suspend_always;
    using std::suspend_never;
}
#else
#include <experimental/coroutine>
namespace codiff {
    using std::experimental::coroutine_handle;
    using std::experimental::suspend_always;
    using std::experimental::suspend_never;
}
#endif


namespace codiff {

template<class T>
struct op;

template<class T>
struct promise
{
    T * m_value = nullptr;

    promise() = default;

    op<T> get_return_object() noexcept
    {
        return { coroutine_handle<promise>::from_promise(*this) };
    }

    constexpr suspend_never initial_suspend() const noexcept { return {}; }
    constexpr suspend_always final_suspend() const noexcept { return {}; }

    suspend_always yield_value(T & value) noexcept
    {
        m_value = std::addressof(value);
        return {};
    }
    suspend_always yield_value(T && value) noexcept { return yield_value(value); }
    // suspend_always yield_value(op<T> & value) noexcept
    // {
    //     return yield_value(value.get());
    // }

    T & value() const noexcept { return *m_value; }

    void unhandled_exception()
    {
        std::rethrow_exception(std::current_exception());
    }
    void return_void() {}
    suspend_never await_transform(auto && value) = delete;
}; // struct promise

template<class T>
struct op
{
    using promise_type = promise<T>;

    coroutine_handle<promise_type> m_coroutine;

    op(coroutine_handle<promise_type> coroutine) noexcept
    :   m_coroutine(coroutine)
    {}
    op(op const& o) = delete;
    op(op && o)
    :   m_coroutine(o.m_coroutine)
    {
        o.m_coroutine = nullptr;
    }

    ~op()
    {
        if(m_coroutine)
        {
            m_coroutine.resume();
            m_coroutine.destroy();
        }
    }

    T& get() const { return m_coroutine.promise().value(); }
    T& operator*() const { return get(); }
    operator T&() const { return get(); }

    // auto & value() const { return get().value(); }
    // auto & grad() const { return get().grad(); }
    // template<class U = var_value_type<T>>
    // void backward(U && dx = 1) { get().backward(dx); }

    // void backward(auto && dx) { get().backward(dx); }
    // void backward() { get().backward(); }
}; // struct op

// template<class T>
// struct var_traits<op<var<T>>>
// {
//     using value_type = T;
// };




template<class T>
auto & value(op<T> const& v)
{ 
    return value(*v);
}

template<class T>
void backward(op<T> const& v, auto && grad)
{
    backward(*v, grad);
}


} // namespace codiff

// TODO namespace properly
template<class T, class... Args>
struct std::experimental::coroutine_traits<codiff::op<T>, Args...>
{
    static_assert(!(std::is_rvalue_reference_v<Args> || ...), "rvalue reference will dangle");

    using promise_type = codiff::promise<T>;
};