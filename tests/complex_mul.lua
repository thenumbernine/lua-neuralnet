#! /usr/bin/env luajit
--[[
Linear approximation to a sine wave based on its previous state.
I guess if I just feed in theta then I'll be asking for a piecewise approximation to a sine wave (and will probably need a big-enough hidden layer?)
If I just feed in sin(theta) or cos(theta) then , for a domain of [0,2pi) I will have multiple outputs for single inputs and therefore not a function.
If I feed in both cos(theta) and sin(theta) then all I'm doing is training the network to converge upon complex multiplication?
--]]

local table = require 'ext.table'

local ANN = require 'neuralnet.ann'
local nn = ANN(2,2)
-- bias just adds more noise
nn.useBias[1] = false
-- so does a non-linear activation
nn.activation = function(x) return x end
nn.activationDeriv = function(x,y) return 1 end
-- with both these off/identity, we are now just gradient-descent converging on complex multiplication
-- so ofc the error goes down to machine precision quickly (in about 1000 iterations)

local phi = math.pi
--local phi = math.pi / 2
--local phi = math.pi / 4
--local phi = math.pi / 8

local results = table()
for i=1,1000 do
	local t = math.random() * 2 * math.pi 
	nn.input[1] = math.cos(t)
	nn.input[2] = math.sin(t)
	nn:feedForward()
	nn.desired[1] = math.cos(t + phi)
	nn.desired[2] = math.sin(t + phi)
	local err = nn:calcError()
	results:insert(err)
	nn:backPropagate(.1)
end

require 'gnuplot'{
	terminal = 'png size 1024,768',
	output = 'complex_mul.png',
	style = 'data points',
	log = 'xy',
	data = {results},
	{using='0:1', title='error'},
}
