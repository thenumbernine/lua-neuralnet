--[[
wrapper class for NeuralNet::ANN / NeuralNetLua.so 's API
the C++ version uses .layers[] .x[] etc

idk maybe I should switch Lua to this?
or maybe I should switch the C++ vectors to be 0-based vectors and then always need a wrapper to abstract it from Lua?

--]]
local assertindex = require 'ext.assert'.index
return function(ctype, nospeedhacks)
	-- right now ctype is the neural net class name, i.e. NeuralNet::ANN<T>
	local ANNctor = assertindex(require 'NeuralNetLua', ctype)
	local Real = assert((ctype:match'^NeuralNet::ANN<(.*)>$'))
	local ANN = setmetatable({}, {
		__call = function(ANN, ...)
			local cppnn = ANNctor(...)

			-- TODO I can't modify ANN.useBatch and have it reflect on all objs ... until I expose static members in the MT
			-- ... but even if I did that, I'd have to separate the static-class-default from the per-network weight ...
			-- ... and there would be no changing of all objs set to the class value (i..e obj field is nil) post-ctor o the obj
			-- so for now just make use of the class wrapper ...
			local nn = {}
			setmetatable(nn, {
				__index = function(t,k)
					if k == 'dt' then return cppnn.dt end
					if k == 'dilution' then return cppnn.dilution end
					if k == 'dropout' then return cppnn.dropout end
					if k == 'useBatch' then return cppnn.useBatch > 0 and cppnn.useBatch or false end
					if k == 'batchCounter' then return cppnn.batchCounter end
					return rawget(nn, k)
				end,
				__newindex = function(t,k,v)
					if k == 'dt' then cppnn.dt = v end
					if k == 'dilution' then cppnn.dilution = v end
					if k == 'dropout' then cppnn.dropout = v end
					if k == 'useBatch' then cppnn.useBatch = v or 0 end
					if k == 'batchCounter' then cppnn.batchCounter = v end
					return rawset(nn, k, v)
				end,
			})

			nn.ptr = cppnn.ptr	-- save the userdata here so the member-function-wrapper cfunctions will think 'nn' is 'cppnn'
			nn.feedForward = cppnn.feedForward
			nn.calcError = cppnn.calcError
			nn.backPropagate = cppnn.backPropagate
			nn.backPropagate_dt = cppnn.backPropagate_dt
			nn.updateBatch = cppnn.updateBatch
			nn.clearBatch = cppnn.clearBatch

			-- how to handle activations ...
			-- should I get rid of the default network activation in Lua?
			-- or should I just convert Lua over to the C++ model already?
			-- until then ...
			-- I'll have to handle read/write to .activation here ...
			-- or I should just replace .activation with :setActivation that sets all layers' activations ...
			function nn:setActivation(name, index)
				if index then
					cppnn.layers[index]:setActivation(name)
				else
					for _,layer in ipairs(cppnn.layers) do
						layer:setActivation(name)
					end
				end
			end
			function nn:setActivationDeriv(name, index)
				if index then
					cppnn.layers[index]:setActivationDeriv(name)
				else
					for _,layer in ipairs(cppnn.layers) do
						layer:setActivationDeriv(name)
					end
				end
			end

			-- instead of redesigning the Lua ANN API, or the code below, I can just make a compat layer within the network table here...
			nn.x = {}
			nn.xErr = {}
			nn.net = {}
			nn.netErr = {}
			nn.w = {}
			nn.dw = {}

			nn.useBias = setmetatable({}, {
				__index = function(t,k)
					return cppnn.layers[k].getBias()
				end,
				__newindex = function(t,k,v)
					cppnn.layers[k]:setBias(v)
				end,
			})

			for i,layer in ipairs(cppnn.layers) do
				nn.x[i] = assert(layer.x)
				nn.xErr[i] = assert(layer.xErr)
				nn.net[i] = assert(layer.net)
				nn.netErr[i] = layer.netErr
				nn.w[i] = assert(layer.w)
				nn.dw[i] = assert(layer.dw)
			end
			nn.x[#cppnn.layers+1] = cppnn.output
			nn.xErr[#cppnn.layers+1] = cppnn.outputError
			-- the downside is the nn structure is still read-only ...

			if nospeedhacks == 'nospeedhacks' then
				nn.input = nn.x[1]
				nn.inputError = nn.xErr[1]
				nn.output = cppnn.output
				nn.outputError = cppnn.outputError
				nn.desired = cppnn.desired
			else
				-- THIS IS THE SLOWDOWN
				-- using the original input as vector-of-Reals, a 1 year summary of nn/prev00000.txt takes 45 seconds
				-- (with pure lua or luajit-ffi it takes 10 seconds)
				-- however
				-- replace nn.input with a cdata ptr and it takes just 5 seconds

				--[=[
				nn.input = nn.layers[1].x
				--]=]
				-- [=[ will this help perf?
				local ffi = require 'ffi'
				local RealPtr = Real..'*'
				require 'ffi.req' 'c.stdlib'	-- strtoul
				local function tocptr(T, uptr)
--DEBUG: print('T', T, 'uptr', uptr)
					local uptrstr = tostring(uptr)
--DEBUG: print('uptrstr', uptrstr)
					local addrstr = uptrstr:match'^userdata: (.*)$'
--DEBUG: print('addrstr', addrstr)
					--local addr = assert(tonumber(addrstr))
					local addr = ffi.C.strtoul(addrstr, ffi.cast('char**', 0), 0)
--DEBUG: print('addr', ('0x%x'):format(addr))
					local ptr = ffi.cast(T, addr)
--DEBUG: print('ptr', ptr)
					return ptr
				end
				--local inputptr = tocptr(RealPtr, cppnn.layers[1].x.v:data())
				--nn.input = inputptr-1	-- make it 1-based
				-- - need to cast to an array type
				-- - need to give the array type a metatable ... with a length oeprator (to trick nn.input , and so nn.input still has native access)
				--[==[ I guess luajit won't let me defined metatables for ctypes that are arrays ... structs only ... and structs don't have overloaded [] operator ...
				local inputlen = #nn.x[1]
				local ffitype = ffi.typeof(Real..'(*)['..inputlen..']')
				-- ... so I can't define the length operator for the nn.input Real* cdata ..
				ffi.metatype(ffitype, {
					__len = function() return inputlen end,
				})
				--]==]
				-- hmmmm should I allow overwriting fields in the C wrapper table ..
				-- at this point why not make a whole new Lua wrapper table ...
				local inputptr = cppnn.layers[1].x.v:data()
				nn.input = tocptr(RealPtr, inputptr) - 1
				nn.inputError = tocptr(RealPtr, cppnn.layers[1].xErr.v:data()) - 1
				nn.output = tocptr(RealPtr, cppnn.output.v:data()) - 1
				nn.outputError = tocptr(RealPtr, cppnn.outputError.v:data()) - 1
				nn.desired = tocptr(RealPtr, cppnn.desired.v:data()) - 1
				--]=]
			end

			local matrix = require 'matrix'
			for _,w in ipairs(nn.w) do
				function w:size()
					return matrix{self:height(), self:width()}
				end
				function w:toLuaMatrix()
					return w:size():lambda(function(i,j) return w[i][j] end)
				end
			end
			for _,x in ipairs(nn.x) do
				function x:toLuaMatrix()
					return matrix{#x}:lambda(function(i) return x[i] end)
				end
			end

			local origNewVector = nn.newVector
			local origNewMatrix = nn.newMatrix
			function nn:newMatrix(...)
				local n = select('#', ...)
				if n == 0 then
					error('expected dimensions')
				elseif n == 1 then
					-- TODO these nils ... I knew it was a bad idea to differentiate between class static member functions and normal functions
					return origNewVector(nil, ...)
				elseif n == 2 then
					return origNewMatrix(nil, ...)
				else
					error("I don't support rank-"..n.." yet")
				end
			end

			return nn
		end,
	})
	return ANN
end
