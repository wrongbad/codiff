
#include <iostream>

#include "gaii/tensor.h"
// #include "gaii/math.h"


using namespace gaii;

int main()
{
    tensor<float, 4> x = 3;
    tensor<float, 1> y = 2;
    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl;
    x += y;
    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl;
    y += x;
    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl;
}