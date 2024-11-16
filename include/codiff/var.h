#pragma once

#include <concepts>

namespace codiff {

template<class T = float>
struct var
{
    // using value_type = T;
    
    T value = 0;
    T grad = 0;
    
    // T const& value() const { return _value; }
    // T & value() { return _value; }
    // T const& grad() const { return _grad; }
    // T & grad() { return _grad; }
    // void backward(T const& dx = 1) { _grad += dx; }
};

template<class T> var(T) -> var<T>;

template<class VarT>
struct var_traits;

template<class T>
struct var_traits<var<T>>
{
    using value_type = T;
};

template<class T>
using var_value_type = typename var_traits<std::remove_cvref_t<T>>::value_type;


template<class T>
T & value(var<T> & v) { return v.value; }

template<class T>
T const& value(var<T> const& v) { return v.value; }

template<class T>
void backward(var<T> & v, auto && grad)
{
    v.grad += grad;
}


template<class T>
requires std::integral<T> || std::floating_point<T>
T value(T const& v) { return v; }

template<class T>
requires std::integral<T> || std::floating_point<T>
int backward(T & v, auto && grad) { return 0; /* no op */ }


template<class T>
concept diffable = requires(T a) // && !std::integral<T> && !std::floating_point<T>
{
    { backward(a, value(a)) } -> std::same_as<void>;
};



} // namespace codiff