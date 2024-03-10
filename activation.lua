-- contains lists of f(x), deriv(x,y), hi, lo

local function tanhDeriv(x,y) return 1 - y * y end

local function poorQuadraticTanh(x)
	return x < -2 and -1 
	or x < 0 and x * (1 + .25*x)
	or x < 2 and x * (1 - .25*x)
	or 1
end

local function poorQuadraticTanhDeriv(x,y)
	return x < -2 and 0
	-- aka `y/x + .25*x` if you like division
	or x < 0 and 1 + .5 * x
	or x < 2 and 1 - .5 * x
	or 0
end

-- [[ useful maybe? tanh cubic approximated across [-2.5,0],[0,2.5]
-- it has a pertty significant error compared to tanh
-- but my goal wasn't replication of tanh,
-- my goal was linear near 0, bounded to [-1, 1], and cubic-continuous
-- it is slightly below tanh(x) on the interval (0, 2.5-epsilon)
local function poorCubicTanh(x)
	return x < -2.5 and -1
	or x < 0 and x * (1 + x * (0.32 + x * 0.032))
	or x < 2.5 and x * (1 + x * (-0.32 + x * 0.032))
	or 1
end
local function poorCubicTanhDeriv(x,y)
	return x < -2.5 and 0
	or x < 0 and 1 + x * (.64 + x * .096)
	or x < 2.5 and	1 + x * (-.64 + x * .096)
	or 0
end
--]]
--[[ TODO 3-part tanh, [-x, -.5],[-.5,.5],[.5, x]
-- but for x=2.5 it goes a lot above tanh(x) and even above 1 ...
-- TODO minimize integral(tanh(x) - cubic(x), x=x1 to x2) ...
--]]

--[[
array contains:
1 = f(x)
2 = df/dx(x,y)
3 = high signal
4 = low signal
--]]
return {
	tanh = {math.tanh, tanhDeriv, -.9, .9, 'math.tanh(x)', '1 - y * y'},
	poorCubicTanh = {poorCubicTanh, poorCubicTanhDeriv, -.9, .9},
	poorQuadraticTanh = {poorQuadraticTanh, poorQuadraticTanhDeriv, -.9, .9},

	-- in case you want to access the funcs by name
	funcs = {
		tanh = math.tanh,
		tanhDeriv = tanhDeriv,
		poorQuadraticTanh = poorQuadraticTanh,
		poorQuadraticTanhDeriv = poorQuadraticTanhDeriv,
		poorCubicTanh = poorCubicTanh,
		poorCubicTanhDeriv = poorCubicTanhDeriv,
	},
}
