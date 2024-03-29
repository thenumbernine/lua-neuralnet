#!/usr/bin/env luajit
local class = require 'ext.class'
local table = require 'ext.table'
local string = require 'ext.string'
local path = require 'ext.path'
local TDNN = require 'neuralnet.tdnn'

math.randomseed(os.time())

local minMaxs = table()
local fn = 'rlnn/tictactoe-minmax.txt'

-- return value is 0-based: 0= none, 1=player 1, 2 = player 2
-- place index is 1-based
local function getPlaceForIndex(board, i)
	return math.floor((board-1) / (3 ^ (i-1))) % 3
end
-- x,y are 1-based
local function getPlace(board, x, y)
	local index = (x-1) + 3 * (y-1) + 1
	return getPlaceForIndex(board, index)
end
local function gameIsDone(board)
	-- check for 3 in a row's
	for player=1,2 do
		if getPlace(board,1,1) == player and getPlace(board,2,2) == player and getPlace(board,3,3) == player then return player end
		if getPlace(board,3,1) == player and getPlace(board,2,2) == player and getPlace(board,1,3) == player then return player end
		for k=1,3 do
			if getPlace(board,k,1) == player and getPlace(board,k,2) == player and getPlace(board,k,3) == player then return player end
			if getPlace(board,1,k) == player and getPlace(board,2,k) == player and getPlace(board,3,k) == player then return player end
		end
	end
	for i=1,9 do
		if getPlaceForIndex(board,i) == 0 then return end	-- return nil, there's an empty place
	end
	return 0	-- all is full, return 0
end
local function scoreForPlayer(player)
	return assert(({
		[0] = 0,	-- ties get 0
		1,			-- x's are first and get a 1
		-1,			-- o's are second and get a -1
	})[player])
end
local function getMoves(board)
	local moves = table()
	for i=1,9 do
		if getPlaceForIndex(board, i) == 0 then moves:insert(i) end
	end
	return moves
end
local function applyMove(move, board, player)
	if getPlaceForIndex(board, move) ~= 0 then
		error("tried to move somewhere someone already had")
	end
	board = board + player * 3 ^ (move - 1)
	return board
end
local function charForPlayer(p)
	if p < 0 or p > 2 then
		error("got unknown place "..tostring(p))
	end
	return ({
		[0] = '.',
		[1] = 'X',
		[2] = 'O',
	})[p]
end
local function printBoard(board)
	print('state',board)
	for i=1,3 do
		for j=1,3 do
			local place = getPlace(board,i,j)
			io.write(charForPlayer(place)..' ')
		end
		print()
	end
	print()
end

if path(fn):exists() then
	minMaxs = string.split(assert(path(fn):read()), '\n'):map(function(l) return tonumber(l) end)
else
	print('building minmax...')	
	local minMaxTotals = {}
	for i=1,3^9 do
		minMaxTotals[i] = 0
		minMaxs[i] = 0
	end
	local function buildState(board, player, moveStates)
		moveStates[player]:insert(1, board)
		-- see if it is terminal
		local winningPlayer = gameIsDone(board)
		if winningPlayer then
			local score = scoreForPlayer(winningPlayer)
			for movePlayer,moveStatesForPlayer in ipairs(moveStates) do
				for _,moveState in ipairs(moveStatesForPlayer) do
					if winningPlayer ~= 1 then score = 0 end
--					if winningPlayer == movePlayer then
--						score = 1 
--					elseif winningPlayer ~= 0 then
--						score = -1
--					else
--						score = 0
--					end
					minMaxs[moveState] = minMaxs[moveState] + score
					minMaxTotals[moveState] = minMaxTotals[moveState] + 1 
				end
			end
		else
			local nextPlayer = 3 - player
			local moves = getMoves(board)
			for _,move in ipairs(moves) do
				local newBoard = applyMove(move, board, player)
				buildState(newBoard, nextPlayer, moveStates)
			end
		end
		moveStates[player]:remove(1)
	end
	buildState(1, 1, table{table(), table()})
	print('...done building minmax')
	for i=1,3^9 do
		if minMaxTotals[i] > 0 then
			minMaxs[i] = minMaxs[i] / minMaxTotals[i]
		end
	end
	local dir = path(fn):getdir()
	if dir.path ~= '.' then
		dir:mkdir()
	end
	assert(path(fn):write(minMaxs:concat('\n')))
end

local board
local nn
local nnTwoLayer = true	-- attempting to make use of neural net optimizations over the tables used for q-learning 
if nnTwoLayer then
	nn = TDNN(9,9,9)	-- input is -1 for O, 0 for empty, +1 for X.  output is which place to play. (but what about X's turn vs O's turn?)
	--nn = TDNN(27,9,3)	-- input: {empty, X, O} times 9 places.  output: stalemate, X wins, O wins
else
	nn = TDNN(3^9,9)	-- this should just rebuild the min/max table.
	nn.nn.useBias[1] = false
end
nn.historySize = 10	-- at least for each move in the board
nn.lambda = 1	-- lambda = 1 means this should operate equivalent to min-max, right?
nn.noise = 0
-- [[	 use linear activation and clear weights
nn.nn.activation = function(x) return x end
nn.nn.activationDeriv = function() return 1 end
for k=1,#nn.nn.w do
	for i=1,#nn.nn.w[k] do
		for j=1,#nn.nn.w[k][i] do
			nn.nn.w[k][i][j] = 0
		end
	end
end
--]]	

-- occlude invalid actions
nn.feedForwardForState = function(self, state)
	-- 2-layer backprop
	if nnTwoLayer then
		for i=1,9 do
			nn.nn.input[i] = -scoreForPlayer(getPlaceForIndex(state, i))
		end
	else
	-- node per state
		for i=1,#nn.nn.input do nn.nn.input[i] = 0 end
		nn.nn.input[state] = 1
	end
	self.nn:feedForward()
end
nn.getBestAction = function(self, qs)
	local occludedQs = qs:map(function(q,i)
		if getPlaceForIndex(board,i) ~= 0 then
			return -math.huge	-- TODO why are these actions getting chosen?
		else
			--print('q move',i,'score',q)
			return q
		end
	end)
	-- picks randomly from all top actions
	local bestAction = TDNN.getBestAction(self, occludedQs)
	-- picks the first best
	--local bestAction = select(2, occludedQs:sup())
	return bestAction
end


local Player = class()

Player.name = 'abstract player' 

function Player:startGame() end

function Player:play(board)
	erorr("TODO implement me!")
end

function Player:done() end


local MinMaxPlayer = class(Player)

MinMaxPlayer.name = 'minmax'
	
function MinMaxPlayer:play(board)
	local moves = getMoves(board)
	local scoresForMoves = moves:map(function(move)
		local scoreForMove = minMaxs[applyMove(move, board, self.playerno)]
		--print('minmax move',move,'score',scoreForMove)
		return scoreForMove
	end)
	for i=1,3 do
		for j=1,3 do
			local move = (i-1) + 3 * (j-1) + 1
			if getPlaceForIndex(board, move) == 0 then
				local newBoard = applyMove(move, board, self.playerno)
				io.write((' %.2f'):format(minMaxs[newBoard]))
			else
				io.write('  *  ')
			end
		end
		print()
	end
	local bestIndex = select(2, scoresForMoves:sup())
	if not bestIndex then
		print('failed to find move')
		print('board',board)
		print('board minmax',minMaxs[board])
		print('moves',moves:concat(','))
		error('')
	end
	local move = moves[bestIndex]
	board = applyMove(move, board, self.playerno)
	return board
end


local NNPlayer = class(Player)

NNPlayer.name = 'qnn'

function NNPlayer:startGame()
	nn.playerno = self.playerno
	self.firstturn = true
end
	
function NNPlayer:play(board)
	if self.firstturn then
		self.firstturn = nil
	else
		nn:applyReward(board, 0)
	end
	
	local action 
	while true do
		action = nn:determineAction(board)
		if getPlaceForIndex(board, action) == 0 then break end
		nn:applyReward(board, 0)	-- TODO hmm , how about a negative reward of some sort?  that means multiple outputs
	end
	
	board = applyMove(action, board, self.playerno)
	
	return board
end
	
function NNPlayer:done(winningPlayer)
	if winningPlayer == 0 then
		nn:applyReward(board, 0)
	elseif winningPlayer ~= self.playerno then
		nn:applyReward(board, -1)
	elseif winningPlayer == self.playerno then
		nn:applyReward(board, 1)
	end
end


local HumanPlayer = class(Player)
HumanPlayer.name = 'human'

function HumanPlayer:play(board)
	printBoard(board)
	local move
	while true do
		while true do
			io.write('>')
			move = io.read'*l'
			print('got move and move', move)
			move = tonumber(move)
			if move and move >= 1 and move <= 9 then break end
			print('not a valid input!')
		end
		move = ({
			[7] = 1,
			[4] = 2,
			[1] = 3,
			[8] = 4,
			[5] = 5,
			[2] = 6,
			[9] = 7,
			[6] = 8,
			[3] = 9,
		})[move]
		if getPlaceForIndex(board, move) == 0 then break end
		print'already a piece there!'
	end

	board = applyMove(move, board, self.playerno)
	return board
end

local RandomPlayer = class(Player)
RandomPlayer.name = 'random'
function RandomPlayer:play(board)
	local move 
	repeat
		move = math.random(9)
	until getPlaceForIndex(board, move) == 0 
	board = applyMove(move, board, self.playerno)
	return board
end

--[[
--]]
local players = {
	--NNPlayer(),
	--MinMaxPlayer(),
	--MinMaxPlayer(),
	--HumanPlayer(),
	--HumanPlayer(),
	RandomPlayer(),
	--RandomPlayer(),
	NNPlayer(),
}
local wins = {[0]=0,0,0}
for iter=1,10000 do
	board = 1
	for playerno,player in ipairs(players) do
		player.playerno = playerno
		player:startGame()
	end
	local playerno = 1
	while true do
		printBoard(board)
		board = players[playerno]:play(board)
		playerno = playerno%#players+1
		
		if gameIsDone(board) then break end
	end
	printBoard(board)

	local winningPlayer = gameIsDone(board)
	
	for _,player in ipairs(players) do
		player:done(winningPlayer)
	end
	
	wins[winningPlayer] = wins[winningPlayer] + 1
	
	if winningPlayer == 0 then
		print('stalemate')
	else
		print(charForPlayer(winningPlayer)..' '..players[winningPlayer].name..' won')
	end

	print('xs '..wins[1]..' os '..wins[2]..' stalemate '..wins[0])
end

