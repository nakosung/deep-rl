{Brain} = require './convnetjs/deepqlearn'
_ = require 'lodash'
fs = require 'fs'
jsonfile = require 'jsonfile'

path = 'network.json'

num_agents = 6

temporal_window = 1
num_inputs = num_agents * 5 - 1
num_actions = 5 + num_agents-1
network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs

layer_defs = []
layer_defs.push type:'input', out_sx:1, out_sy:1, out_depth:network_size
layer_defs.push type:'fc', num_neurons:50, activation:'relu'
layer_defs.push type:'fc', num_neurons:50, activation:'relu'
layer_defs.push type:'regression', num_neurons:num_actions

tdtrainer_options = learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01

opt = 
	temporal_window : temporal_window
	experience_size : 30000
	start_learn_threshold : 1000
	gamma : 0.7
	learning_steps_total : 200000
	learning_steps_burnin : 3000
	epsilon_min : 0.05
	epsilon_test_time : 0.05
	layer_defs : layer_defs
	tdtrainer_options : tdtrainer_options

world_size = 16
screen_size = 20
grid = Math.floor screen_size / world_size
max_hp = 5
cooldown = 5
range = 3

pos = (x) ->
	x / world_size

TermUI = 
	out: (buf) ->
		process.stdout.write(buf)
		this

	clear: ->
		@out '\x1b[2J'
		this

	pos: (x, y) ->
		@out "\x1b[#{y};#{x}H"
		this

	fg: (c) ->
		@out "\x1b[3#{c}m"
		this

class Agent
	constructor : (@world,@brain,@team) ->
		@x = Math.floor(Math.random() * world_size) + 1
		@y = Math.floor(Math.random() * world_size) + 1
		@hp = max_hp
		@actions = [ [1,0], [-1,0], [0,1], [0,-1], 'nothing' ]
		@digestion_signal = 0
		@reward = 0
		@cooldown = 0
		@dead = false
		@id = @world.alloc_id()

	forward : ->
		enemies = @world.enemy(@)
		input_array = [pos(@x),pos(@y),@hp/max_hp,@cooldown/cooldown]
		for enemy in enemies
			input_array = input_array.concat [pos(enemy.x-@x),pos(enemy.y-@y),enemy.hp/max_hp,enemy.cooldown/cooldown,Math.abs(@team - enemy.team)]
		@action = @brain.forward input_array

	backward : ->
		@brain.backward @reward
		@reward = 0

	distance : (enemy) ->
		dx = enemy.x - @x
		dy = enemy.y - @y
		[dx,dy]

	take_damage : (attacker) ->
		@world.log "#{@id} attacked by #{attacker?.id}"
		@hp -= 1
		if @hp > 0
			if attacker?
				attacker.reward += 1
			@reward -= 2
		else
			@die(attacker)

	die : (attacker) ->
		@world.log "#{@id} killed by #{attacker?.id}"
		if attacker?
			attacker.hp = max_hp
			attacker.reward += 10
		@reward -= 20
		for agent in @world.agents
			if agent != @ and agent != attacker
				if agent.team == @team
					agent.reward -= 5
				else if attacker?
					agent.reward += 5
		@dead = true

	heal : (healer) ->
		if @hp == max_hp
			healer?.reward -= 10
		else
			@world.log "#{@id} healed by #{healer?.id}"
		
			@hp = Math.min(max_hp,@hp+1)
			@reward += 1
			healer?.reward += 5

	skill : (enemy) ->
		return if enemy.dead

		if enemy.team == @team
			enemy.heal(@)
		else
			enemy.take_damage(@)

	tick : ->
		# if @x * 2 == world_size and @y * 2 == world_size
		# 	@heal()

		if @action < num_agents-1
			enemy = @world.enemy(@)[@action]
			[dx,dy] = @distance enemy
			sq_dist = dx*dx + dy*dy
			sq_range = range*range
			if sq_dist > sq_range or @cooldown > 0
				@reward -= 10 # cannot do that
			else
				@cooldown = cooldown

				if Math.random() < (sq_dist - 1) / sq_range
					@skill enemy
					@world.log "#{@id} got critical"
					@skill enemy
				else
					@skill enemy
					
		else
			@cooldown-- if @cooldown > 0

			action = @actions[@action - num_agents + 1]

			if action == 'nothing'

			else
				[dx,dy] = action
				old_x = @x
				old_y = @y
				@x = Math.max(1,Math.min(world_size,@x + dx))
				@y = Math.max(1,Math.min(world_size,@y + dy))
				if old_x == @x and old_y == @y
					@reward -= 10 # cannot do that

	dump : ->
		TermUI.pos(@x*grid,@y*grid).fg(@team+5).out("#{@id}").pos(grid*@x,grid*@y+1).out("#{@hp}/#{@cooldown}")
	
class World
	constructor : ->
		@next_id = 0
		@brain = new Brain(num_inputs, num_actions, opt)
		
		@agents = []

		@spawn(x&1) for x in [1..num_agents]
		
		@clock = 0
		@logs = []
		@load(path) if fs.exists(path)
		

	log : (x...) ->
		@logs.push x.join(' ')
		@logs.shift() if @logs.length > 20

	alloc_id : ->
		@next_id++

	spawn : (team) ->
		@agents.push new Agent(@,@brain,team)

	enemy : (me) ->
		enemies = _.filter @agents, (x) -> x != me
		_.sortBy enemies, (x) ->
			offset = if x.team == me.team then 0 else 100000	
			dx = x.x - me.x
			dy = x.y - me.y
			offset + dx * dx + dy * dy

	tick : ->
		@clock++

		for agent in @agents
			agent.forward()

		for agent in @agents
			agent.tick()

		for agent in @agents
			agent.backward()

		N = @agents.length
		deads = _.filter @agents, (x) -> x.dead
		if deads.length
			@agents = _.filter @agents, (x) -> not x.dead
			for dead in deads
				@spawn(dead.team)
			

		@dump() if @clock > opt.learning_steps_burnin / 2 or @clock % 5 == 0 
		@save(path) if @clock % 100 == 0
		#@quake() if @clock % 100 == 0

	quake : ->
		agent.take_damage() for agent in @agents

	dump : ->
		TermUI.clear()
		for agent in @agents
			agent.dump()
		TermUI.fg(7)
		@logs.map (log,k) ->
			TermUI.pos(40,k).out(log)
		TermUI.pos(0, grid*world_size+2).out("clock:#{@clock}").pos(0,grid*world_size+4)

	save : (file) ->
		jsonfile.writeFileSync(file, @brain.value_net.toJSON())

	load : (file) ->
		json = jsonfile.readFileSync(file)
		@brain.value_net.fromJSON(json)
		
world = new World()
while true
	world.tick() 