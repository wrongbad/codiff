#include "gaii/math.h"

#include <iostream>

using namespace gaii;

struct Linear
{
    var<> w = {1.1};
    var<> b = {2};

    op<var<>> operator()(var<> & in)
    {
        co_yield in * w + b;
    }
};

struct RNN
{
    var<> w = {0.5};

    using Output = std::tuple<var<>&, var<>&>;

    op<Output> operator()(var<> & in, var<> & state)
    {
        auto new_state = state + (in - state) * w;
        auto out = in - new_state;
        co_yield {out, new_state};
    }
};

struct Model
{
    Linear l0;
    Linear l1;
    RNN l2;

    using Output = std::tuple<var<float>&, var<float>&>;

    op<Output> operator()(var<float> & in, var<float> & state)
    {
        auto x0 = l0(in);
        auto x1 = l1(x0);
        auto x2 = l2(x1, state);
        auto &[out, new_state] = *x2;
        co_yield {out, new_state};
    }
};


int main(int argc, char** argv)
{
    var a = {2.0f};
    var state = {0.0f};
    Model m;

    {
        auto outs = m(a, state);
        auto &[out, new_state] = *outs;
        out.backward(1);
        std::cout << out.value() << std::endl;
        std::cout << out.grad() << std::endl;
        state = new_state;
    }

    // std::cout << m.l1.w.grad() << std::endl;
    std::cout << m.l0.b.grad() << std::endl;
    std::cout << m.l0.w.grad() << std::endl;
    std::cout << a.grad() << std::endl;

    return 0;
}