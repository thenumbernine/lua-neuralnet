[![Donate via Stripe](https://img.shields.io/badge/Donate-Stripe-green.svg)](https://buy.stripe.com/00gbJZ0OdcNs9zi288)<br>
[![Donate via Bitcoin](https://img.shields.io/badge/Donate-Bitcoin-green.svg)](bitcoin:37fsp7qQKU8XoHZGRQvVzQVP8FrEJ73cSJ)<br>
[![Donate via Paypal](https://img.shields.io/badge/Donate-Paypal-green.svg)](https://buy.stripe.com/00gbJZ0OdcNs9zi288)

Some neural network models I was messing with in Lua

- ANN = artifical neural network.  multi-layer, back-propagation.
- QNN = q-learning extension applied to an ANN.
- TDNN = td-learning extension.

## tests:
- xor.lua = 2-layer MLP backpropagation learning of the XOR problem.
- cartpole.lua = reinforcement learning via QNN or TDNN to the cart-pole balancing problem.
- randomwalk.lua = reinforcement learning of the random walk problem.

tests that are still in the works:
- tictactoe.lua = tic tac toe neural net vs min/max algorithm
- poker.lua = pokerbot 5000

requires
- [lua-ext](https://github.com/thenumbernine/lua-ext)
- [lua-matrix](https://github.com/thenumbernine/lua-matrix)
- [lua-gl](https://github.com/thenumbernine/lua-gl) for some tests
