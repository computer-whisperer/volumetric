Here is the basic idea:

Suppose we define a physical 3d model as a simple wasm function. This function will take in a 3d point and return a simple boolean indicating whether the point is inside the model or not.

This is attractive for a few reasons, but the main potential flaw is that sampling the model will be expensive. The question is, how slow? Given the reality of many projects that could benefit from a simpler model definition paradigm, and how cheap compute is in reality, is this viable?


Therefore, here we will be creating the MVP. We will need to divide this into two parts: a wasm module that defines the model (currently should be defined in crates/test_model) and a root module that will a) render the model and b) convert the model to various other 3d formats.