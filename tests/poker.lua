require 'ext'
local TDNN = require 'neuralnet.tdnn'

math.randomseed(os.time())

function table.giveTo(from, to)
	while #from > 0 do
		to:insert(from:remove())
	end
end

local function range(a,b,c)
	local t = table()
	if not c then
		for x=a,b do 
			t:insert(x)
		end
	else
		for x=a,b,c do
			t:insert(x)
		end
	end
	return t
end


local PokerCard = class()

function PokerCard:init(value, suit)
	self.value = value
	self.suit = suit
end

function PokerCard:__tostring()
	return self:numberStr() .. self:suitStr()
end

function PokerCard.__concat(a,b) return tostring(a) .. tostring(b) end

function PokerCard:numberStr()
	return ({
		[13] = 'A',
		[12] = 'K',
		[11] = 'Q',
		[10] = 'J',
		[9] = 'T',
	})[self.value] or tostring(self.value+1)
end

local charHeart = 'h'
local charSpade = 's'
local charDiamond = 'd'
local charClub = 'c'

PokerCard.suits = {charHeart, charSpade, charDiamond, charClub}

function PokerCard:suitStr()
	return self.suits[self.suit]
end

function PokerCard.__eq(a,b) 
	return a.suit == b.suit and a.value == b.value
end

local function pokerScore(...)
	local hand = table()
	for _,part in ipairs{...} do
		hand:append(part)		-- don't modify the source parameters
	end
	assert(#hand == 5)
	
	hand:sort(function(a,b) return a.value > b.value end)	-- highest value first 
	local flush = hand[1].suit == hand[2].suit
	and hand[2].suit == hand[3].suit
	and hand[3].suit == hand[4].suit
	and hand[4].suit == hand[5].suit
	local straight = (hand[1].value == hand[2].value-1
				and hand[2].value == hand[3].value-1
				and hand[3].value == hand[4].value-1
				and hand[4].value == hand[5].value-1)
		or (hand[5].value == 13	-- aces can be low for straights
				and hand[4].value == 4
				and hand[3].value == 3
				and hand[2].value == 2
				and hand[1].value == 1)
	local counts = range(1,13):map(function() return 0 end)
	for _,card in ipairs(hand) do
		counts[card.value] = counts[card.value] + 1
	end
	local pairs = counts:map(function(count,value,tbl)
		if count > 1 then
			return {value=assert(value), count=assert(count)}, #tbl+1
		end
	end)
	pairs:sort(function(a,b)
		return a.count > b.count	-- highest count first
	end)
	--print(pairs:map(function(v) return v.value..'x'..v.count end):concat(' '))

	local score
	if straight and flush then
		score = 8	-- straight flush
		if hand[1].value == 13 and hand[2].value == 4 then 
			score = score + (4-1)/13
		else
			score = score + (hand[1].value-1)/13
		end
	elseif #pairs == 1 and pairs[1].count == 4 then
		score = 7	-- four of a kind
		score = score + (hand[1].value-1) / 13 + (hand[5].value-1) / (13*13)
	elseif #pairs == 2 and pairs[1].count == 3 and pairs[2].count == 2 then
		score = 6	-- full house
		score = score + (hand[1].value-1) / 13 + (hand[4].value-1) / (13*13)
	elseif flush then
		score = 5	-- flush
		-- how many high cards are matched before the pot is split?
		score = score + (hand[1].value-1) / 13 + (hand[2].value-1) / (13*13) + (hand[3].value-1) / (13*13*13) + (hand[4].value-1) / (13*13*13*13)
	elseif straight then
		score = 4	-- straight
		if hand[1].value == 13 and hand[2].value == 4 then 
			score = score + (4-1)/13
		else
			score = score + (hand[1].value-1)/13
		end
	elseif #pairs == 1 and pairs[1].count == 3 then
		score = 3	-- three of a kind
		score = score + (hand[1].value-1) / 13 + (hand[4].value-1) / (13*13) + (hand[5].value-1) / (13*13*13)
	elseif #pairs == 2 and pairs[1].count == 2 and pairs[2].count == 2 then
		score = 2	-- two pair
		score = score + (hand[1].value-1) / 13 + (hand[3].value-1) / (13*13) + (hand[5].value-1) / (13*13*13)
	elseif #pairs == 1 and pairs[1].count == 2 then
		score = 1	-- pair
		score = score + (hand[1].value-1) / 13 + (hand[3].value-1) / (13*13) + (hand[4].value-1) / (13*13*13) + (hand[5].value-1) / (13*13*13*13)
	else
		score = (hand[1].value-1) / 13 + (hand[2].value-1) / (13*13) + (hand[3].value-1) / (13*13*13) + (hand[4].value-1) / (13*13*13*13) + (hand[5].value-1) / (13*13*13*13*13)
	end
	return score, hand
end

local PokerDeck = class()

function PokerDeck:init()
	self.cards = table()
	for i=1,13 do	-- 1 is 2, 13 is Ace, so it is sequential and 1-based
		for j=1,4 do
			self.cards:insert(PokerCard(i,j))
		end
	end
end

function PokerDeck:shuffle()
	local newCards = table()
	while #self.cards > 0 do
		newCards:insert(self.cards:remove(math.random(#self.cards)))
	end
	self.cards = newCards
end

local Player = class()
function Player:init()
	self.cards = table()
end
function Player:cantDoThat() end
function Player:action() return 0 end

-- bpn cheater
local PokerNN = class(Player)

--[[
what do we want to represent?
first, our two cards ... 
next, the remaining, and whether they're visible
--]]
function PokerNN:init()
end

local PlayerHuman = class(Player)

function PlayerHuman:action(up)
	print('what do you do?')
	print('[c]all, [r]aise, [f]old')
	local result
	local raise
	repeat
		result = io.read('*l')
		if result:sub(1,1) == 'r' then
			raise = tonumber(result:sub(2):trim())
			result = 'r'
		end
	until result and (result == 'c' or result == 'r' or result == 'f')
	if result == 'f' then return -1  end
	if result == 'c' then return 0 end
	
	print('raise how much?')
	while not (raise and raise > 0) do	
		raise = tonumber(io.read('*l'))
	end
	return raise
end
function PlayerHuman:cantDoThat()
	print("you can't do that")
end

local PlayerMonteCarlo = class(Player)
function PlayerMonteCarlo:action(up) 
	local deck = PokerDeck()
	for _,card in ipairs(self.cards) do
		deck.cards:remove(deck.cards:find(card))
	end
	for _,card in ipairs(up) do
		deck.cards:remove(deck.cards:find(card))
	end
	
	local wins = 0
	local losses = 0
	local total = 10000
	local fakeUp = table()
	local fakeOpponent = table()
	for iter=1,total do
		assert(#deck.cards == 52 - #self.cards - #up)
		deck:shuffle()
		
		local fakeOtherHand = table()
		while #fakeOtherHand < 2 do
			fakeOtherHand:insert(deck.cards:remove())
		end
		
		while #fakeUp < 3 - #up do
			assert(#deck.cards > 0)
			fakeUp:insert(deck.cards:remove())
		end
		
		local myFakeScore = pokerScore(up, fakeUp, self.cards)
		local otherFakeScore = pokerScore(up, fakeUp, fakeOtherHand)
		if myFakeScore > otherFakeScore then
			wins = wins + 1
		else
			losses = losses + 1
		end
		
		while #fakeUp > 0 do deck.cards:insert(fakeUp:remove()) end
		while #fakeOtherHand > 0 do deck.cards:insert(fakeOtherHand:remove()) end
	end

	local winchance = wins / total
	print('chance of winning with '..self.cards:map(function(s) return tostring(s) end):concat(' ')..' is '..(100*winchance)..'%')

	if winchance >= .5 then return 50 end
	if winchance < .3 then return -1 end	-- TODO progressively less as the #up increases?
	return 0
end

function testPoker()
	local players = table{
		PlayerHuman(),
		PlayerMonteCarlo(),
	}
	local blind = 5
	for _,player in ipairs(players) do
		player.money = 1000
	end
	while true do
		local deck = PokerDeck()
		local up = table()
		
		deck:shuffle()
		-- deal cards to players
		for i=1,2 do
			for _,player in ipairs(players) do
				player.cards:insert(assert(deck.cards:remove()))
			end
		end
		
		local playerTotals = table()	-- total (minus current round) on the table so far
		for i=1,#players do
			playerTotals[i] = 0
		end
		local playerFolds = table()	-- which players have folded
		for i=1,#players do
			playerFolds[i] = false
		end
		local playerBets = table()	-- bets for the current round
		for i=1,#players do
			playerBets[i] = 0
		end
		for flops=0,3 do
			-- commence one round of bidding ... 
			local playerIndex	-- ... or start at the best hand
			for i=1,#players do
				if not playerFolds[i] then
					playerIndex = i
					break
				end
			end
			assert(playerIndex)	-- we should never start a game with everyone folded
			local lastPlayerToRaise = playerIndex
			local currentBet = 0
			repeat
				print()
				for i,player in ipairs(players) do
					io.write('player '..i)
					if getmetatable(player) == PlayerHuman then
						io.write(' hand: '..player.cards:map(function(c) return tostring(c) end):concat(' '))
					end
					io.write(' money '..player.money)
					print()
				end
				print('dealt: '..up:map(function(c) return tostring(c) end):concat(' '))
				print('player '..playerIndex..'...')
			
				if not playerFolds[playerIndex] then
					local tryAgain
					local totalToRaise
					local raise
					repeat
						tryAgain = nil
						raise = players[playerIndex]:action(up)	-- return -1 for fold, 0 for call, positive for bet
						if raise == -1 then
							print('player '..playerIndex..' folds')
							playerFolds[playerIndex] = true
						else
							if raise == 0 then
								print('player '..playerIndex..' calls')
							else
								print('player '..playerIndex..' raises '..raise)
							end
							local upToBet = currentBet - playerBets[playerIndex]
							totalToRaise = upToBet + raise
							if totalToRaise > players[playerIndex].money then	-- tried to give too much 
								players[playerIndex]:cantDoThat()	-- penalty for doing something stupid
								tryAgain = true	-- then go again
							end
						end
					until not tryAgain
					if not playerFolds[playerIndex] then
						players[playerIndex].money = players[playerIndex].money - totalToRaise
						playerBets[playerIndex] = playerBets[playerIndex] + totalToRaise
						if raise > 0 then
							currentBet = currentBet + raise
							lastPlayerToRaise = playerIndex
						end
					end
				end
				-- considering the player may have folded, count the total number of players remaining
				-- if there's only one player left then they win
				local playersLeft = players:filter(function(player,i) return playerFolds[i] end)
				if #playersLeft == 1 then break end
				
				playerIndex = playerIndex % #players + 1
			until playerIndex == lastPlayerToRaise
			
			-- if we had a winner then break out
			local playersLeft = players:filter(function(player,i) return playerFolds[i] end)
			if #playersLeft == 1 then break end

			-- otherwise push all bets to totals (into the pot)
			for i=1,#players do
				playerTotals[i] = playerTotals[i] + playerBets[i]
				playerBets[i] = 0
			end

			if flops < 3 then 
				-- and turn another card over
				local card = assert(deck.cards:remove())
				up:insert(card)
				print()			
				print('revealing '..card)
			end
		end
		local playersLeft = players:filter(function(player,i) return playerFolds[i] end)
		local winners = table()
		if #playersLeft == 1 then
			winners:insert(1)
		else
			for i,player in ipairs(players) do
				if not playerFolds[i] then
					local score, hand = pokerScore(player.cards, up)
					print('player',i,'hand',hand:map(function(c) return tostring(c) end):concat(' '),'score',score)
					winners:insert{index = i, score = score}
				end
			end

			winners:sort(function(a,b) return a.score > b.score end)	-- highest first

			local i = 1
			while i <= #winners and winners[i].score == winners[1].score do
				i = i + 1
			end
			i = i - 1
			-- now all players >= i are losing
			while #winners > i do winners:remove() end

			winners = winners:map(function(w) return w.index end)
		end

		print('#winners',#winners)
		print('winner indexes: ',  unpack(winners))
		
		-- now divide the money among the winners
		local pot = playerTotals:sum()
		while #winners > 0 do	
			local cut = math.floor(pot / #winners)
			local winner = players[winners:remove()]
			assert(cut <= pot)
			pot = pot - cut
			winner.money = winner.money + cut
		end

		print('done!')
		print()

		-- now give the cards back
		up:giveTo(deck.cards)
		for i,player in ipairs(players) do
			player.cards:giveTo(deck.cards)
		end
		assert(#deck.cards == 52)
	end
end

testPoker()


