#pragma once

#include "gaii/var.h"
#include "gaii/op.h"

#include <cmath>

namespace gaii {

template<class T>
T && fwd(auto && x) { return std::forward<T>(x); }

template<class A>
using unary_op = op<var<std::remove_cvref_t<decltype(value(std::declval<A>()))>>>;

template<class A, class B>
using binary_op = op<var<decltype(value(std::declval<A>()) * value(std::declval<B>()))>>;



template<diffable A>
auto operator-(A && a)
{
    return [] (A a) -> unary_op<A> {
        var y {-value(a)};
        co_yield y;
        backward(a, -y.grad);
    }(fwd<A>(a));
}

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
        // std::cout << "op-:b " << y.grad << std::endl;
        backward(a, y.grad);
        backward(b, -y.grad);
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
        backward(a, y.grad * y.value);
    }(fwd<A>(a));
}

template<diffable A>
auto tanh(A && a)
{
    return [] (A a) -> unary_op<A> {
        using std::tanh;
        var y {tanh(value(a))};
        co_yield y;
        backward(a, y.grad * y.value * (1 - y.value));
    }(fwd<A>(a));
}

template<diffable A>
auto sigmoid(A && a)
{
    return [] (A a) -> unary_op<A> {
        var y {sigmoid(value(a))};
        co_yield y;
        backward(a, y.grad * (1 - y.value * y.value));
    }(fwd<A>(a));
}




template<diffable A>
auto logsumexp(A && a)
{
    return [] (A a) -> op<var<decltype(logsumexp(value(a)))>> {
        auto & x = value(a);
        auto xmax = broadcast<1>(x, [] (auto & xi) { return max(xi); });
        auto xexp = exp(x - xmax);
        auto sumexp = broadcast<1>(xexp, [] (auto & xi) { return sum(xi); });
        var y {log(sumexp) + xmax};
        co_yield y;
        // std::cout << "logsumexp:b " << y.grad << std::endl;
        backward(a, y.grad * xexp / sumexp);
    }(fwd<A>(a));
}



template<diffable A>
auto log_softmax(A && a)
{
    return [] (A a) -> unary_op<A> {
        auto y = a - logsumexp(a);
        co_yield y;
        // std::cout << "log_softmax:b " << y->grad << std::endl;
    }(fwd<A>(a));
}



} // namespace gaii