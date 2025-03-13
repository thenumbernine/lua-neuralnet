local ANN = require 'neuralnet.ann'
local class = require 'ext.class'
local table = require 'ext.table'

-- q-learner with NN impl
local QNN = class()

QNN.gamma = .9
QNN.alpha = .5
QNN.noise = .1

function QNN:init(...)
	local layerSizes = {...}
	self.nn = ANN(...)
	-- TODO disable all biases?
	-- TODO clear initial weights?
	-- TODO use linear activation function?
end

--[[
This function is responsible for translating the state into neuron input signals.
This is the one function that has to be replaced if you are going to change state from an integer to anything else.
--]]
function QNN:feedForwardForState(state)
	for i=1,#self.nn.input do
		self.nn.input[i] = i == state and 1 or 0
		-- TODO high/low signal values?
	end
	self.nn:feedForward()
end

function QNN:getQs(state)
	self:feedForwardForState(state)
	return setmetatable({unpack(self.nn.output)}, table)
end

-- function QNN:getReward(state, action, nextState)
-- function QNN:applyAction(state, action)
function QNN:getBestAction(qs)
	if math.random() < self.noise then
		return math.random(#qs)
	else
		local best = table()
		for i=1,#qs do
			if #best == 0 or qs[i] > qs[best[1]] then
				best = table{i}
			elseif qs[i] == qs[best[1]] then
				best:insert(i)
			end
		end
		-- TODO or number weighting or first-best
		return best:pickRandom()
	end
end

function QNN:determineAction(state)
	-- S[t] is our current state, represented as 'state'

	-- A[t] is our action for this state.  get it by getting the Q's of our current state, permuting them sightly, and picking the highest action
	local thisQs = self:getQs(state)					-- Q(S[t], *)

	self.lastState = state								-- S[t]
	self.lastAction = self:getBestAction(thisQs)		-- A[t]
	self.lastStateActionQ = thisQs[self.lastAction]		-- Q(S[t], A[t])

	return self.lastAction, self.lastStateActionQ
end

function QNN:applyReward(newState, reward)--, lastState, lastAction, lastStateActionQ)
	local nextQs = self:getQs(newState)		-- Q(S[t+1], *)
	local maxNextQ = nextQs:sup()			-- max(Q(S[t+1], *))

	-- setup input for backpropagation
	self:feedForwardForState(self.lastState)
	for i=1,#self.nn.outputError do
		self.nn.outputError[i] = 0
	end
	local err = reward + self.gamma * maxNextQ - self.lastStateActionQ
	self.nn.outputError[self.lastAction] = err
	self.nn:backPropagate(self.alpha)

	return err
end

function QNN:__tostring()
	local qs = table()
	for i=1,#self.nn.input do
		local lq, rq = unpack(self:getQs(i))
		lq = ('%.2f'):format(lq)
		rq = ('%.2f'):format(rq)
		local q = lq..'/'..rq
		qs:insert(q)
	end
	return qs:concat(' ')
end

function QNN.__concat(a,b) return tostring(a) .. tostring(b) end

return QNN
