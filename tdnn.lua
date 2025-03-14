local QNN = require 'neuralnet.qnn'
local class = require 'ext.class'
local table = require 'ext.table'

local TDNN = class(QNN)

TDNN.historySize = 10
TDNN.lambda = .7

function TDNN:init(...)
	TDNN.super.init(self, ...)
	self.history = table()
end

function TDNN:determineAction(state, ...)
	local action, actionQ = TDNN.super.determineAction(self, state, ...)
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

function TDNN:applyReward(state, reward, ...)
	local err = TDNN.super.applyReward(self, state, reward, ...)

	for i=2,#self.history do
		-- damp the first -- it has already been applied in QNN:applyReward
		err = err * self.lambda

		local history = self.history[i]
		self:feedForwardForState(history.state)
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

return TDNN
