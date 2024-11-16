#pragma once

#include "codiff/var.h"
#include "codiff/op.h"

#include <cmath>

namespace codiff {

template<class T>
T && fwd(auto && x) { return std::forward<T>(x); }

template<class A>
using unary_op = op<var<std::remove_cvref_t<decltype(value(std::declval<A>()))>>>;

template<class A, class B>
using binary_op = op<var<decltype(value(std::declval<A>()) * value(std::declval<B>()))>>;




template<class A, class B>
requires diffable<A> || diffable<B>
auto operator+(A && a, B && b)
{
    return [] (A a, B b) -> binary_op<A, B> {
        var y {value(a) + value(b)};
        co_yield y;
        backward(a, y.grad);
        backward(b, y.grad);
    }(fwd<A>(a), fwd<B>(b));
}

template<class A, class B>
requires diffable<A> || diffable<B>
auto operator-(A && a, B && b)
{
    return [] (A a, B b) -> binary_op<A, B> {
        var y {value(a) - value(b)};
        co_yield y;
        backward(a, y.grad);
        backward(a, -y.grad);
    }(fwd<A>(a), fwd<B>(b));
}

template<class A, class B>
requires diffable<A> || diffable<B>
auto operator*(A && a, B && b)
{
    return [] (A a, B b) -> binary_op<A, B> {
        var y {value(a) * value(b)};
        co_yield y;
        backward(a, value(b) * y.grad);
        backward(b, value(a) * y.grad);
    }(fwd<A>(a), fwd<B>(b));
}

template<class A, class B>
requires diffable<A> || diffable<B>
auto operator%(A && a, B && b)
{
    return [] (A a, B b) -> op<var<decltype(value(a) % value(b))>> {
        var y {value(a) % value(b)};
        co_yield y;
        backward(a, mat_mul<false, true>(y.grad, value(b)));
        backward(b, mat_mul<true, false>(value(a), y.grad));
    }(fwd<A>(a), fwd<B>(b));
}

template<diffable A>
auto exp(A && a)
{
    return [] (A a) -> unary_op<A> {
        using std::exp;
        var y {exp(value(a))};
        co_yield y;
        backward(a, value(y) * y.grad);
    }(fwd<A>(a));
}

template<diffable A>
auto tanh(A && a)
{
    return [] (A a) -> unary_op<A> {
        using std::tanh;
        var y {tanh(value(a))};
        co_yield y;
        backward(a, value(y) * (1 - value(y)) * y.grad);
    }(fwd<A>(a));
}

template<diffable A>
auto sigmoid(A && a)
{
    return [] (A a) -> unary_op<A> {
        var y {sigmoid(value(a))};
        co_yield y;
        backward(a, (1 - value(y) * value(y)) * y.grad);
    }(fwd<A>(a));
}



} // namespace codiff