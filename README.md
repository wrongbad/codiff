# codiff

Eager autograd with c++20 coroutines - the esoteric library no one has been asking for!

GAII design (Gradient Acquisition Is Initialization)

True to the philosophy of c++, it's now possible to auto-diff modular code without requiring dynamic memory allocation

Since the backward pass is fully compile time code, we get compiler optimizations for free

Bonus tensor library included with template based numpy-style broadcasting

# How does it work?

Every calculation that depends on a `var<T>` is a coroutine.

Each coroutine runs immediately to its first yield point, which yields a `var<T>` containing the value of the calculation, and a placeholder for the gradient.

Each coroutine captures its inputs, so the full computation graph is held implicitly this way.

When you've computed your loss functions or something, you can set the gradient with `backward()`. 

Then all you need to do is let the output variable fall out of scope.

The coroutine destructor will resume the coroutine, which will execute any manual backward pass code.

Then it will destroy all the local variable and temporaries (including nested coroutines) in reverse construction order, which propages the gradients backward perfectly. 

GAII in action!