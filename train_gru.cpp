#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

#include <codiff/tensor.h>
#include <codiff/math.h>

using codiff::tensor;
using codiff::var;
using codiff::op;

struct RNG
{
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist {0.0, 1.0};
    operator float() { return dist(rng); }
};

tensor<RNG> fill;

template<int Ni, int No>
struct GRU
{
    var<tensor<float, Ni, No>> w_x_z {fill};
    var<tensor<float, Ni, No>> w_x_r {fill};
    var<tensor<float, Ni, No>> w_x_h {fill};
    var<tensor<float, No, No>> w_h_z {fill};
    var<tensor<float, No, No>> w_h_r {fill};
    var<tensor<float, No, No>> w_h_h {fill};
    var<tensor<float, No>> b_z {0};
    var<tensor<float, No>> b_r {0};
    var<tensor<float, No>> b_h {0};

    template<int... B>
    op<var<tensor<float, B..., No>>> fwd(
        var<tensor<float, B..., Ni>> & x, var<tensor<float, B..., No>> & h)
    {
        auto z = sigmoid(x % w_x_z + h % w_h_z + b_z);
        auto r = sigmoid(x % w_x_r + h % w_h_r + b_r);
        auto h2 = sigmoid(x % w_x_h + (h * r) % w_h_h + b_h);
        co_yield (1 - z) * h + z * h2;
    }
};



int main()
{
    std::stringstream ss;
    ss << std::ifstream("data/alice.txt").rdbuf();
    std::string train = ss.str();

    std::cout << train.size() << std::endl;

    GRU<256, 256> gru;
    var<tensor<float, 1, 256>> state {0};

    for(char c : train)
    {
        var<tensor<float, 1, 256>> input {0};
        input.value(0, uint8_t(c)) = 1;
        auto out = gru.template fwd<1>(input, state);
        std::cout << value(out) << std::endl;

        break;
    }
}