{Brain} = require './convnetjs/deepqlearn'
_ = require 'lodash'
fs = require 'fs'
jsonfile = require 'jsonfile'
argv = require('minimist')(process.argv.slice(2))

path = 'network.json'

num_agents = 4

temporal_window = 2
world_size = 4
num_channels = 3 * 2
num_inputs = world_size * world_size * num_channels
num_actions = 5 + num_agents-1
network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs

layer_defs = []
layer_defs.push type:'input', out_sx:1, out_sy:1, out_depth:network_size
layer_defs.push type:'fc', num_neurons:70, activation:'relu'
layer_defs.push type:'fc', num_neurons:70, activation:'relu'
layer_defs.push type:'regression', num_neurons:num_actions

tdtrainer_options = learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01

speed_factor = argv.speed or 1

opt = 
	temporal_window : temporal_window
	experience_size : 30000 / speed_factor
	start_learn_threshold : 1000 / speed_factor
	gamma : 0.85
	learning_steps_total : 200000 / speed_factor
	learning_steps_burnin : 3000 / speed_factor
	epsilon_min : 0.05
	epsilon_test_time : 0.05
	layer_defs : layer_defs
	tdtrainer_options : tdtrainer_options

screen_size = 20
grid = Math.floor screen_size / world_size
max_hp = 10
cooldown = 2
cooldown_heal = 20
deadbody_sleep = if num_agents > 2 then 1 else 1
range = 3
range_heal = 3

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

input_array = [1..world_size*world_size*3].map -> 0

reset_input = ->
	input_array[i-1] = 0 for i in [1..world_size*world_size*num_channels]

write_input = (x,y,ch,val=1) ->
	input_array[(x-1 + (y-1) * world_size) * num_channels + ch] = val

class Agent
	actions : [ [1,0], [-1,0], [0,1], [0,-1], 'nothing' ]

	constructor : (@world,@brain,@team) ->
		@y = (world_size-1) * @team
		@x = world_size >> 1
		@hp = max_hp		
		@vel = [0,0]
		@reward = 0
		@cooldown_heal = 0
		@cooldown = 0
		@dead = false
		@id = @world.alloc_id()
		@dead_counter = 0

	forward : ->
		return if @is_dead()

		enemies = @world.enemy(@)
		
		reset_input()
		write_input(@x,@y,0,@hp/max_hp)
		write_input(@x,@y,1,@cooldown/cooldown)
		write_input(@x,@y,2,@cooldown_heal/cooldown_heal)
		for enemy in enemies
			unless enemy.is_dead()
				page = Math.abs(enemy.team-@team) + 1
				write_input(enemy.x,enemy.y,3*page,enemy.hp/max_hp)
				write_input(enemy.x,enemy.y,3*page+1,enemy.cooldown/cooldown)
				write_input(enemy.x,enemy.y,3*page+2,enemy.cooldown_heal/cooldown_heal)
		@action = @brain.forward input_array

	backward : ->
		return if @is_dead()

		@brain.backward @reward
		@reward = 0

	distance : (enemy) ->
		dx = enemy.x - @x
		dy = enemy.y - @y
		[dx,dy]

	name : ->
		"#{@team}:#{@id}"

	rel_log : (subject,rel) ->
		@world.log "#{subject?.name()} #{rel} #{@name()}"

	take_damage : (attacker) ->
		@rel_log attacker, "-"
		@hp -= 1
		if @hp > 0
			if attacker?
				attacker.reward += 10
		else
			@hp = 0
			@die(attacker)

	is_dead : -> @hp == 0

	die : (attacker) ->
		@rel_log attacker, "X"
		if attacker?
			@world.scores[attacker.team]++
			attacker.heal()
			attacker.reward += 20
		else
			@reward -= 5
		@reward -= 5
		for agent in @world.agents
			if agent != @ and agent != attacker
				if agent.team == @team
					agent.reward -= 5
				else if attacker?
					agent.reward += 10
		@die_counter = deadbody_sleep

	heal : (healer) ->
		if @hp == max_hp
			healer?.reward -= 100
		else
			@rel_log healer, "+"
		
			@hp = Math.min(max_hp,@hp+2)
			@reward += 0.1

	skill : (enemy) ->
		if enemy.team == @team
			if @cooldown_heal > 0 or not @check_dist(enemy,range)
				@reward -= 100
			else
				@cooldown_heal = cooldown_heal
				enemy.heal(@)
		else
			if @cooldown > 0 or not @check_dist(enemy,range_heal)
				@reward -= 100
			else
				enemy.take_damage(@)
				@cooldown = cooldown

	check_dist : (target,range) ->
		[dx,dy] = @distance target
		dx*dx + dy*dy <= range * range

	tick : ->
		if @die_counter > 0
			if --@die_counter == 0
				@dead = true
			return

		old_vel = @vel
		@vel = [0,0]

		# if @x * 2 == world_size and @y * 2 == world_size
		# 	@heal()

		if @action < num_agents-1
			enemy = @world.enemy(@)[@action]
			if enemy.is_dead()
				@reward -= 100
				return 
			
			@skill enemy
					
		else
			@cooldown-- if @cooldown > 0
			@cooldown_heal-- if @cooldown_heal > 0

			action = @actions[@action - num_agents + 1]

			if action == 'nothing'

			else
				@vel = [dx,dy] = action
				old_x = @x
				old_y = @y
				@x = Math.max(1,Math.min(world_size,@x + dx))
				@y = Math.max(1,Math.min(world_size,@y + dy))
				if old_x == @x and old_y == @y
					@reward -= 100 # cannot do that

		# 관성이 좋아요!
		dx = @vel[0] - old_vel[0] 
		dy = @vel[1] - old_vel[1]
		@reward -= (dx * dx + dy * dy) * 0.1

		dx = (@x - world_size / 2) / world_size
		dy = (@y - world_size / 2) / world_size

		@reward += Math.max( 0, 1.0/4 - (dx * dx + dy * dy) ) * 4 - 2

		@reward += @hp / max_hp - 0.5

	color : ->
		TermUI.fg(@team+5)

	dump : ->
		@color()
		TermUI.pos(@x*grid,@y*grid).out("#{@id}").pos(grid*@x,grid*@y+1)
		text = []
		if @die_counter > 0
			text.push "X"
		else
			if @hp < max_hp
				text.push "H:#{@hp - max_hp}"
			if @cooldown
				text.push "C:#{@cooldown}"
			if @cooldown_heal
				text.push "h:#{@cooldown_heal}"
		if text.length
			TermUI.out text.join('/')
		
class World
	constructor : ->
		@next_id = 0
		@scores = [0,0]
		@brains = [1..num_agents].map -> new Brain(num_inputs, num_actions, opt)		
		@free_brains = @brains.slice()

		@learning = not argv.run

		@clock = 0
		@logs = []				

		@brains_for_teams = [@free_brains,@free_brains]
		@brainpool_for_teams = [@brains,@brains]
		unless @learning			
			@trained_brains = [1..num_agents].map -> new Brain(num_inputs, num_actions, opt)		
			@load(path,@trained_brains) 			
			brain.learning = false for brain in @trained_brains		
			@free_trained_brains = @trained_brains.slice()
			@brains_for_teams[0] = @free_trained_brains		
			@brains_for_teams[1] = @free_trained_brains		

		
		@agents = []

		@spawn(x&1) for x in [1..num_agents]
		
		

	log : (x...) ->
		@logs.push x.join(' ')
		@logs.shift() if @logs.length > 20

	alloc_id : ->
		@next_id++

	spawn : (team) ->
		brain = @brains_for_teams[team].pop()		
		@agents.push new Agent(@,brain,team)

	enemy : (me) ->
		enemies = _.filter @agents, (x) -> x != me
		_.sortBy enemies, (x) ->
			offset = if x.team == me.team then 0 else 100000	
			dx = x.x - me.x
			dy = x.y - me.y
			offset + dx * dx + dy * dy

	tick : ->
		@clock++

		agent.forward() for agent in @agents			
		agent.tick() for agent in @agents
		agent.backward() for agent in @agents			

		N = @agents.length
		deads = _.filter @agents, (x) -> x.dead
		if deads.length
			@agents = _.filter @agents, (x) -> not x.dead
			for dead in deads
				@brains_for_teams[dead.team].push dead.brain
				@spawn(dead.team)
			
		
		#@quake() if @clock % 100 == 0

		@dump() if not @learning or @clock > opt.learning_steps_burnin or @clock % 5 == 0 
		@save(path) if @learning and @clock % 5000 == 0

	quake : ->
		for agent in @agents
			agent.take_damage()  if Math.random() < 0.5				

	dump : ->
		TermUI.clear()
		for agent in @agents
			agent.dump()
		TermUI.fg(7)
		@logs.map (log,k) ->
			TermUI.pos(40,k).out(log)
		TermUI.pos(0, grid*world_size+2).out("clock:#{@clock} score:#{@scores.join(':')}").pos(0,grid*world_size+4)
		@brains.map (brain,k) ->
			TermUI.pos(0,grid*world_size+3+k).out("eps:#{brain.epsilon} age:#{brain.age}, loss:#{brain.average_loss_window.get_average()}, s-reward:#{brain.average_reward_window.get_average()}")

	save : (file) ->
		return
		jsonfile.writeFileSync("#{team}-#{file}", brain.value_net.toJSON()) for brain,team in @brains
		@log 'network saved'

	load : (file,brains) ->
		brain.value_net.fromJSON(jsonfile.readFileSync("#{team}-#{file}")) for brain,team in brains
		@log 'network loaded'
		
world = new World()
if world.learning
	while true
		world.tick() 
else		
	setInterval world.tick.bind(world), 0
