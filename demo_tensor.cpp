#include "codiff/math.h"
#include "codiff/tensor.h"

#include <iostream>

using namespace codiff;

using dtype = tensor<float, 1>;

struct Linear
{
    var<dtype> w = {1.1};
    var<dtype> b = {2};

    op<var<dtype>> operator()(var<dtype> & in)
    {
        co_yield in * w + b;
    }
};

struct RNN
{
    var<dtype> w = {0.5};

    using Output = std::tuple<var<dtype>&, var<dtype>&>;

    op<Output> operator()(var<dtype> & in, var<dtype> & state)
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

    using Output = std::tuple<var<dtype>&, var<dtype>&>;

    op<Output> operator()(var<dtype> & in, var<dtype> & state)
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
    var<dtype> a = {2.0f};
    var<dtype> state = {0.0f};
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


    tensor<float, 2> x = {1, 2};
    tensor<float, 2> y = x + 3;
    std::cout << "y = " << y << std::endl;

    return 0;
}