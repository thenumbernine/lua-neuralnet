local ANN = require 'neuralnet.ann'
local class = require 'ext.class'
local table = require 'ext.table'
local string = require 'ext.string'
local math = require 'ext.math'

-- q-learner with NN impl
local QEnv = class()

QEnv.gamma = .9
QEnv.alpha = .5
QEnv.noise = .1

function QEnv:init(...)
	-- should nn go here or in agent?
	self.nn = ANN(...)
	-- TODO disable all biases?
	-- TODO clear initial weights?
	-- TODO use linear activation function?
end

function QEnv:feedForwardForState(agent, state)
	agent:observe(state)	-- convert 'state' into nn inputs
	self.nn:feedForward()
end

-- function QEnv:getReward(state, lastState)
-- function QEnv:applyAction(state, action)
function QEnv:determineAction(agent, state, noise)
	-- determine our best action on the environment
	-- requires the actions to be discrete
	self:feedForwardForState(agent, state)
	local qs = self.nn.output
--print(table.concat(qs, ', '))

	noise = noise or self.noise or 0
	local bestAction = 1
	--local bestActions = setmetatable({1}, table)
	local bestValue = qs[1] + math.random() * noise
--DEBUG:assert(math.isfinite(bestValue))
	for i=2,#qs do
		local checkValue = qs[i] + math.random() * noise
--DEBUG:assert(math.isfinite(checkValue))
		if bestValue < checkValue then
			bestValue = checkValue
			bestAction = i
			--bestActions:insert(i)
		end
	end
	--local bestAction = bestActions:pickRandom()
	return bestAction, qs[bestAction]
end

function QEnv:applyReward(agent, newState, reward, lastState, lastAction, lastStateActionQ)
	local nn = self.nn
-- [[ setup input for backpropagation
-- requires actions to be discrete
	local maxNextQ = select(2, self:determineAction(agent, newState, 0))			-- max(Q(S[t+1], *))
	local err = reward + self.gamma * maxNextQ - lastStateActionQ
	self:feedForwardForState(agent, lastState)
	for i=1,#nn.outputError do
		nn.outputError[i] = 0
	end
	nn.outputError[lastAction] = err
--]]
--[[ if the output is y_i = delta_ik for lastAction=k
-- https://stats.stackexchange.com/a/187747
-- nabla_w Q(s,a) = dQ/dw_ij ... Q is a vector ...
-- y_i = sigma(w_ij x_j) ...
-- dy/dw_ij = sigma'(w_ij x_j) * x_j
-- 'outputError' is usually dE/dy_i for a scalar E, but we have no scalar E so ...
-- ... but where is 'lastAction'? anywhere?
-- but it still seems to work just fine.
	local maxNextQ = select(2, self:determineAction(agent, newState, 0))			-- max(Q(S[t+1], *))
	local err = reward + self.gamma * maxNextQ - lastStateActionQ
	self:feedForwardForState(agent, lastState)
	for i=1,#nn.outputError do
		nn.outputError[i] = nn.output[i] * err
	end
--]]

	nn:backPropagate(self.alpha)

	return err
end

--[[
agent provides:
	:resetState()
	:performAction()
	:getReward()
--]]
function QEnv:step(agent, state)
	-- state = S[t] is our current state

	-- determine next action.
	local action, actionQ = self:determineAction(agent, state)		-- A[t], Q(S[t], A[t])
	-- action = A[t] is our action for this state.  get it by getting the Q's of our current state, permuting them sightly, and picking the highest action

	local newState = agent:performAction(state, action, actionQ)

	-- determine reward and whether to reset
	local reward, reset = agent:getReward(newState, state)

	--apply reward
	-- applies reward with action as the A(S[t],*) and actionQ
	self:applyReward(agent, newState, reward, state, action, actionQ)

	if reset then
		self.history = table()
		newState = agent:resetState()
	end

	return newState, reward, reset
end

function QEnv:__tostring()
	local qsForSs = table()
	for i,w in ipairs(self.nn.w) do
		qsForSs:insert(require 'ext.tolua'(w))
	end
	return qsForSs:concat(' ')
end

QEnv.__concat = string.concat

return QEnv
