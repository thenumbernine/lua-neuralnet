local QEnv = require 'neuralnet.qnn'
local class = require 'ext.class'
local table = require 'ext.table'

local TDEnv = class(QEnv)

TDEnv.historySize = 10
TDEnv.lambda = .7

function TDEnv:init(...)
	TDEnv.super.init(self, ...)
	self.history = table()
end

function TDEnv:determineAction(agent, state, ...)
	local action, actionQ = TDEnv.super.determineAction(self, agent, state, ...)
	-- insert at the beginning
	self.history:insert(1, {
		state=state,
		action=action,
	})
	-- remove from end
	while #self.history > self.historySize do
		self.history:remove()
	end
	return action, actionQ
end

function TDEnv:applyReward(agent, state, reward, ...)
	local err = TDEnv.super.applyReward(self, agent, state, reward, ...)

	for i=2,#self.history do
		-- damp the first -- it has already been applied in QEnv:applyReward
		err = err * self.lambda

		local history = self.history[i]
		self:feedForwardForState(agent, history.state)
		for j=1,#self.nn.outputError do
			self.nn.outputError[j] = 0
		end
		self.nn.outputError[history.action] = err
		self.nn:backPropagate(self.alpha)
		-- now is it the error (change from old to new Q) of the most recent history that is degraded?
		-- or is it the reward that is degraded?
	end

	return err
end

return TDEnv
