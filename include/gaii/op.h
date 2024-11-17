#pragma once

#include "gaii/var.h"

#include <exception>

#if __has_include(<coroutine>)
#include <coroutine>
#define GAII_COROTINE_NAMESPACE std
#else
#include <experimental/coroutine>
#define GAII_COROTINE_NAMESPACE std::experimental
#endif


namespace gaii {

template<class T>
struct op;

template<class T>
struct promise
{
    using suspend_always = GAII_COROTINE_NAMESPACE::suspend_always;
    using suspend_never = GAII_COROTINE_NAMESPACE::suspend_never;
    using coro_handle = GAII_COROTINE_NAMESPACE::coroutine_handle<promise>;

    T * m_value = nullptr;

    promise() = default;

    op<T> get_return_object() noexcept
    {
        return { coro_handle::from_promise(*this) };
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
    using coro_handle = typename promise_type::coro_handle;
    
    coro_handle m_coroutine;

    op(coro_handle coroutine) noexcept
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
    T* operator->() const { return &get(); }
    operator T&() const { return get(); }

    auto & get_value() const { return get().get_value(); }
    auto & get_grad() const { return get().get_grad(); }
    void backward(auto && grad) { get().backward(grad); }
}; // struct op


// template<class T>
// struct var_traits<op<var<T>>>
// {
//     using value_type = T;
//     static const bool diffable = true;
// };


// template<class T>
// auto & value(op<T> const& v)
// { 
//     return value(*v);
// }

// template<class T>
// void backward(op<T> const& v, auto && grad)
// {
//     backward(*v, grad);
// }


} // namespace gaii

template<class T, class... Args>
struct GAII_COROTINE_NAMESPACE::coroutine_traits<gaii::op<T>, Args...>
{
    static_assert(!(std::is_rvalue_reference_v<Args> || ...), "rvalue reference will dangle");

    using promise_type = gaii::promise<T>;
};