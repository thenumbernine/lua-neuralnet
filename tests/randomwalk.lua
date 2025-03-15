#!/usr/bin/env luajit

-- random walk RL
local maxiter = ... or 5

math.randomseed(os.time())

local size = 11
local qnn = require 'neuralnet.tdnn'(size, 2)	qnn.historySize = 100
--local qnn = require 'neuralnet.qnn'(size, 2)
qnn.alpha = .1
qnn.gamma = .99
qnn.noise = 1e-5
qnn.nn.useBias[1] = false
-- [[
qnn.nn.activation = function(x) return x end
qnn.nn.activationDeriv = function() return 1 end
for i=1,size do
	qnn.nn.w[1][1][i] = 0
	qnn.nn.w[1][2][i] = 0
end
--]]
local function printState(state)
	for i=1,size do
		local s = ('%.2f/%2.f'):format(qnn.nn.w[1][1][i], qnn.nn.w[1][2][i])
		if i == state then s = '['..s..']' else s = ' '..s..' ' end
		io.write(' | '..s)
	end
	print()
end

local class = require 'ext.class'
local RandomWalk = class()
function RandomWalk:resetState()
	return math.ceil(size/2)
end
function RandomWalk:observe(state)
	local nn = qnn.nn
	for i=1,#nn.input do
		nn.input[i] = 0
	end
	if 1 <= state and state <= #nn.input then
		nn.input[state] = 1
	end
end
function RandomWalk:performAction(state, action, actionQ)
	if action == 1 then
		state = state-1
	elseif action == 2 then
		state = state+1
	else
		error('got bad action: '..tostring(action))
	end
	return state
end
function RandomWalk:getReward(state)
	local reward = 0
	local reset
	if state == 0 then
		reward = -1
		reset = true
	elseif state == size+1 then
		reward = 1
		reset = true
	end
	return reward, reset
end

local agent = RandomWalk()
qnn.agent = agent
for iter=1,maxiter do
	local numSteps = 0
	local reward, reset
	local state = agent:resetState()
	repeat
printState(state)
		state, reward, reset = qnn:step(agent, state)
		numSteps = numSteps + 1
	until reset
printState(state)
	print(qnn..' steps:'..numSteps)
	print()
end
