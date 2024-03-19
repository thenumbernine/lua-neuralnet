--[[
wrapper class for NeuralNet::ANN / NeuralNetLua.so 's API
the C++ version uses .layers[] .x[] etc

idk maybe I should switch Lua to this?
or maybe I should switch the C++ vectors to be 0-based vectors and then always need a wrapper to abstract it from Lua?

--]]
local assertindex = require 'ext.assert'.index
return function(ctype)
	local ANNctor = assertindex(require 'NeuralNetLua', ctype)
	local ANN = setmetatable({}, {
		__call = function(ANN, ...)
			local nn = ANNctor(...)
			-- instead of redesigning the Lua ANN API, or the code below, I can just make a compat layer within the network table here...
			nn.x = {}
			nn.xErr = {}
			nn.net = {}
			nn.netErr = {}
			nn.w = {}
			nn.dw = {}
			nn.useBias = setmetatable({}, {
				__index = function(t,k)
					return nn.layers[k].getBias()
				end,
				__newindex = function(t,k,v)
					nn.layers[k]:setBias(v)
				end,
			})
			for i,layer in ipairs(nn.layers) do
				nn.x[i] = assert(layer.x)
				nn.xErr[i] = assert(layer.xErr)
				nn.net[i] = assert(layer.net)
				nn.netErr[i] = layer.netErr
				nn.w[i] = assert(layer.w)
				nn.dw[i] = assert(layer.dw)
			end
			nn.x[#nn.layers+1] = nn.output
			nn.xErr[#nn.layers+1] = nn.outputError
			-- the downside is the nn structure is still read-only ...
		
			-- [=[ 
			nn.input = nn.layers[1].x
			--]=]
			--[=[ will this help perf? 
			local ffi = require 'ffi'
			local inputptr = assert(tonumber(tostring(nn.layers[1].x.v:data().ptr):match'^userdata: (.*)$'))
			-- TODO 
			-- - need to cast to an array type
			-- - need to give the array type a metatable ... with a length oeprator (to trick nn.input , and so nn.input still has native access)
			nn.input = ffi.cast('double*', inputptr )-1
			--]=]

			-- TODO I can't modify ANN.useBatch and have it reflect on all objs ... until I expose static members in the MT
			-- ... but even if I did that, I'd have to separate the static-class-default from the per-network weight ... 
			-- ... and there would be no changing of all objs set to the class value (i..e obj field is nil) post-ctor o the obj
			nn.useBatch = ANN.useBatch
			for _,w in ipairs(nn.w) do
				function w:toLuaMatrix()
					return require 'matrix'.lambda({w:height(), w:width()}, function(i,j) return w[i][j] end)
				end
			end

			-- how to handle activations ...
			-- should I get rid of the default network activation in Lua?
			-- or should I just convert Lua over to the C++ model already?
			
			-- until then ...
			-- I'll have to handle read/write to .activation here ...
			-- or I should just replace .activation with :setActivation that sets all layers' activations ...
			function nn:setActivation(name, index)
				if index then
					nn.layers[index]:setActivation(name)
				else
					for _,layer in ipairs(nn.layers) do
						layer:setActivation(name)
					end
				end
			end
			function nn:setActivationDeriv(name, index)
				if index then
					nn.layers[index]:setActivationDeriv(name)
				else
					for _,layer in ipairs(nn.layers) do
						layer:setActivationDeriv(name)
					end
				end
			end
			
			return nn
		end,
	})
	return ANN
end
