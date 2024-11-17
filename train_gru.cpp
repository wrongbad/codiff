#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

#include <gaii/tensor.h>
#include <gaii/math.h>
#include <gaii/optim.h>

#include <fenv.h> 


using gaii::tensor;
using gaii::var;
using gaii::op;

struct RNG
{
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist {-0.5, 0.5};
    operator float() { return dist(rng); }
};

tensor<RNG> fill;


template<class Optimizer, int Nin, int Nout>
struct Linear
{
    template<class T>
    using Param = typename Optimizer::template param<T>;

    Optimizer & opt;
    Param<tensor<float, Nin, Nout>> w {{fill}, opt};
    Param<tensor<float, Nout>> b {{0}, opt};

    op<var<tensor<float, 1, Nout>>> operator()(
        var<tensor<float, 1, Nin>> & x)
    {
        co_yield x % w + b;
    }
};


template<class Optimizer, int Nin, int Nout>
struct GRU
{
    template<class T>
    using Param = typename Optimizer::template param<T>;

    Optimizer & opt;
    Param<tensor<float, Nin, Nout>> w_x_z {{fill}, opt};
    Param<tensor<float, Nin, Nout>> w_x_r {{fill}, opt};
    Param<tensor<float, Nin, Nout>> w_x_h {{fill}, opt};
    Param<tensor<float, Nout, Nout>> w_h_z {{fill}, opt};
    Param<tensor<float, Nout, Nout>> w_h_r {{fill}, opt};
    Param<tensor<float, Nout, Nout>> w_h_h {{fill}, opt};
    Param<tensor<float, Nout>> b_z {{0}, opt};
    Param<tensor<float, Nout>> b_r {{0}, opt};
    Param<tensor<float, Nout>> b_h {{0}, opt};

    op<var<tensor<float, 1, Nout>>> operator()(
        var<tensor<float, 1, Nin>> & x, 
        var<tensor<float, 1, Nout>> & h)
    {
        auto z = sigmoid(x % w_x_z + h % w_h_z + b_z);
        auto r = sigmoid(x % w_x_r + h % w_h_r + b_r);
        auto h2 = sigmoid(x % w_x_h + (h * r) % w_h_h + b_h);
        auto hh = (1 - z) * h + z * h2;
        co_yield hh;
    }
};

template<class Optimizer, int Nin, int Nout, int Nembed>
struct CharModel
{
    Optimizer & opt;
    Linear<Optimizer, Nin, Nembed> w_in {opt};
    GRU<Optimizer, Nembed, Nembed> rnn[2] {{opt}, {opt}};
    Linear<Optimizer, Nembed, Nout> w_out {opt};

    struct Output
    {
        var<tensor<float, 1, Nout>> & out;
        var<tensor<float, 1, Nembed>> & h0;
        var<tensor<float, 1, Nembed>> & h1;
    };
    op<Output> operator()(
        var<tensor<float, 1, Nin>> & x,
        var<tensor<float, 1, Nembed>> & h0,
        var<tensor<float, 1, Nembed>> & h1)
    {
        auto x0 = w_in(x);
        auto r0 = rnn[0](x0, h0);
        auto x1 = x0 + r0;
        auto r1 = rnn[1](x1, h1);
        auto x2 = x1 + r1;
        auto o = w_out(x2);
        co_yield {o, r0, r1};
    }
};

template<int Steps>
void train_batch(char const* inputc, char const* targetc, auto & model, auto & h0, auto & h1, bool print)
{
    static float logp_avg = -10;

    if constexpr ( Steps > 0 )
    {
        var<tensor<float, 1, 256>> input {0};
        input.value(0, uint8_t(*inputc)) = 1;

        var<tensor<float, 1, 256>> target {0};
        target.value(0, uint8_t(*targetc)) = 1;

        auto outs = model(input, h0, h1);
        auto &[out, h0_next, h1_next] = *outs;

        auto lm = log_softmax(out);
        lm.backward(-target.value); // NLL loss
        
        float logp = value(lm)(0, uint8_t(*targetc));
        logp_avg += (logp - logp_avg) * 0.001;
        if(print)
            std::cout << logp_avg << std::endl;

        train_batch<Steps-1>(inputc+1, targetc+1, model, h0_next, h1_next, false);
    }
}


int main(int argc, char ** argv)
{
    std::stringstream ss;
    ss << std::ifstream("data/alice.txt").rdbuf();
    std::string train = ss.str();

    std::cout << train.size() << std::endl;

    // // feenableexcept(FE_INVALID | FE_OVERFLOW);
    // fenv_t fenv;
    // unsigned int new_excepts = (FE_INVALID | FE_OVERFLOW) & FE_ALL_EXCEPT;
    // if ( fegetenv(&fenv) ) { return -1; }
    // unsigned int old_excepts = fenv.__control & FE_ALL_EXCEPT;
    // fenv.__control &= ~new_excepts;
    // fenv.__mxcsr   &= ~(new_excepts << 7);
    // if ( fesetenv(&fenv) ) { return -1; }

    gaii::optim::sgd opt { .lr = 0.0003 };
    // gaii::optim::adam opt {
    //     .lr = 0.0001,
    //     .beta1 = 0.9,
    //     .beta2 = 0.999,
    // };

    CharModel<decltype(opt), 256, 256, 64> model {opt};

    constexpr int Nchunk = 8;

    gaii::var<tensor<float, 1, 64>> h[2] = {{0}, {0}};

    char const* trainc = train.c_str();
    for(int offset = 0 ; offset + Nchunk < train.size() ; offset += Nchunk)
    {
        bool print = (offset / Nchunk) % 100 == 0;

        train_batch<Nchunk>(trainc+offset, trainc+offset+1, model, h[0], h[1], print);

        opt.step ++;
    }
}