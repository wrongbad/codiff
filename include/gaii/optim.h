#pragma once

namespace gaii {
namespace optim {


struct sgd
{
    float lr = 0.0003;
    float grad_clamp = 1;
    float param_clamp = 5;
    int step = 0;

    template<class T>
    struct param : var<T>
    {
        sgd & opt;
        int step = 0;

        void backward(auto && grad)
        {
            auto & g = this->grad;
            if(step != opt.step)
            {
                clamp_inplace(grad, -opt.grad_clamp, opt.grad_clamp);
                this->value -= opt.lr * grad;
                clamp_inplace(this->value, -opt.param_clamp, opt.param_clamp);
                step = opt.step;
            }
            g += grad;
        }
    };
};

struct adam
{
    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    float grad_clamp = 1;
    float param_clamp = 5;
    int m1mass = 1 / (1 - beta1);
    int m2mass = 1 / (1 - beta2);
    int step = 0;
    
    template<class T>
    struct param : var<T>
    {
        adam & opt;
        T m1 = 0;
        T m2 = 0;
        int step = 0;

        void backward(auto && grad)
        {
            auto & g = this->grad;

            if(step != opt.step)
            {
                clamp_inplace(g, -opt.grad_clamp, opt.grad_clamp);

                m1 += (g - m1) / std::min(opt.m1mass, step);
                m2 += (g*g - m2) / std::min(opt.m2mass, step);

                this->value -= opt.lr * m1 / (sqrt(m2) + opt.eps);

                clamp_inplace(this->value, -opt.param_clamp, opt.param_clamp);

                step = opt.step;
            }

            g += grad;
        }
    };
};



} // namespace optim
} // namespace gaii