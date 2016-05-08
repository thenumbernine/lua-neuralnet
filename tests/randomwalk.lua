#!/usr/bin/env luajit

-- random walk RL
local maxiter = ... or 5

local TDNN = require 'neuralnet.tdnn'

math.randomseed(os.time())

local size = 11
local tdnn = TDNN(size, 2)
tdnn.noise = 0
tdnn.nn.useBias[1] = false
-- [[	
tdnn.nn.activation = function(x) return x end
tdnn.nn.activationDeriv = function() return 1 end
for i=1,size do
	tdnn.nn.w[1][1][i] = 0
	tdnn.nn.w[1][2][i] = 0
end
--]]	
local function applyAction(state, action)
	if action == 1 then return state-1 end
	if action == 2 then return state+1 end
	error('idk')
end
local function printState(state)
	for i=1,size do
		local s = ('%.2f/%2.f'):format(tdnn.nn.w[1][1][i], tdnn.nn.w[1][2][i])
		if i == state then s = '['..s..']' end
		io.write(' '..s)
	end
	print()
end
local gamma = .9
local alpha = 1
for iter=1,maxiter do
	local state = math.ceil(size/2)
	local numSteps = 0
	repeat
		--printState(state)
		local action = tdnn:determineAction(state)
		state = applyAction(state, action)
		local reward = ({[0]=-1, [size+1]=1})[state] or 0
		tdnn:applyReward(state, reward)
		numSteps = numSteps + 1
	until state == 0 or state == size+1
	--printState(state)
	print(tdnn..' steps:'..numSteps)
	--print()
end
