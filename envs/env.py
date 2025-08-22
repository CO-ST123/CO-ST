"""
Traffic network simulator w/ defined sumo files
@author: Tianshu Chu
"""
import logging
import random

import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci_v as traci
import xml.etree.cElementTree as ET
import os


DEFAULT_PORT = 4050
SEC_IN_MS = 1000

REALNET_REWARD_NORM = 20

class PhaseSet:
    def __init__(self, phases):
        self.num_phase = len(phases)
        # self.num_signal = len(signals)
        # self.num_speed = len(speeds)
        self.num_lane = len(phases[0])
        self.phases = phases
        # self.signals = signals
        # self.speeds = speeds
        # self._init_phase_set()

    @staticmethod
    def _get_phase_lanes(phase, signal='r'):
        phase_lanes = []
        for i, l in enumerate(phase):
            if l == signal:
                phase_lanes.append(i)
        return phase_lanes

    def _init_phase_set(self):
        self.red_lanes = []
        # self.green_lanes = []
        for phase in self.phases:
            self.red_lanes.append(self._get_phase_lanes(phase))
            # self.green_lanes.append(self._get_phase_lanes(phase, signal='G'))

class SpeedSet:
    def __init__(self, speeds):
        # self.num_phase = len(phases)
        # self.num_signal = len(signals)
        self.num_speed = len(speeds)
        # self.num_lane = len(phases[0])
        # self.phases = phases
        # self.signals = signals
        self.speeds = speeds
        # self._init_phase_set()


class SignalSet:
    def __init__(self, signals):
        self.num_signal = len(signals)
        self.signals = signals
        # self._init_phase_set()

class PhaseMap:
    def __init__(self):
        self.phases = {}
        self.signals = {}
        self.speeds = {}

    def get_phase(self, phase_id, action):
        return self.phases[phase_id].phases[int(action)]

    def get_speed(self, speed_id, action):
        return self.speeds[speed_id].speeds[int(action)]

    def get_signal(self, signal_id, action):
        return self.signals[signal_id].signals[int(action)]

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase

    def get_signal_num(self, signal_id):
        return self.signals[signal_id].num_signal

    def get_speed_num(self, speed_id):
        return self.speeds[speed_id].num_speed

    def get_lane_num(self, phase_id):
        return self.phases[phase_id].num_lane

    def get_red_lanes(self, phase_id, action):
        return self.phases[phase_id].red_lanes[int(action)]


class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control
        # self.edges_in = []
        self.lanes_in = []
        self.ilds_in = []
        self.segs_in = []
        self.fingerprint = []
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0
        self.num_fingerprint = 0
        self.wave_state = []
        self.wait_state = []
        # self.waits = [] 
        self.phase_id = -1
        self.speed_id = -1
        self.n_a = 0
        self.n_a_p = 0
        self.n_a_s = 0
        self.prev_action = -1


class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        self.name = config.get('scenario')
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]
        self._init_map()
        self.init_data(is_record, record_stats, output_path)
        self.init_test_seeds(test_seeds)
        self._init_sim(self.seed)
        self._init_nodes()

        self.n_episode = -1
        self.start_time = time.time()
        self.pre_time = time.time()

        self.terminate()

    def _debug_traffic_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase = self.sim.trafficlight.getRedYellowGreenState(self.node_names[0])
            cur_traffic = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'node': node_name,
                           'action': node.prev_action,
                           'phase': phase}
            for i, ild in enumerate(node.ilds_in):
                cur_name = 'lane%d_' % i
                cur_traffic[cur_name + 'queue'] = self.sim.lane.getLastStepHaltingNumber(ild)
                cur_traffic[cur_name + 'flow'] = self.sim.lane.getLastStepVehicleNumber(ild)
                # cur_traffic[cur_name + 'wait'] = node.waits[i]
            self.traffic_data.append(cur_traffic)

    def _get_node_phase(self, action, node_name, phase_type):
        node = self.nodes[node_name]
        cur_phase = self.phase_map.get_phase(node.phase_id, action)
        if phase_type == 'green':
            return cur_phase
        prev_action = node.prev_action
        node.prev_action = action
        if (prev_action < 0) or (action == prev_action):
            return cur_phase
        prev_phase = self.phase_map.get_phase(node.phase_id, prev_action)
        switch_reds = []
        switch_greens = []
        for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
            if (p0 in 'Gg') and (p1 == 'r'):
                switch_reds.append(i)
            elif (p0 in 'r') and (p1 in 'Gg'):
                switch_greens.append(i)
        if not len(switch_reds):
            return cur_phase
        yellow_phase = list(cur_phase)
        for i in switch_reds:
            yellow_phase[i] = 'y'
        for i in switch_greens:
            yellow_phase[i] = 'r'
        return ''.join(yellow_phase)

    def _get_vehicle_speed(self, action, node_name, vehicle_id):
        node = self.nodes[node_name]
        cur_speed = self.phase_map.get_speed(node.speed_id, action)
        speed = self.sim.vehicle.getSpeed(vehicle_id)

        for a in cur_speed:
            if a == 'a':
                speed_adv = speed + 1.39
            elif a == 'd':
                speed_adv = speed - 1.39
                if speed_adv < 0:
                    speed_adv = 0.1
            else:
                speed_adv = speed

        return speed_adv

    def _get_Hvehicle_speed(self, action, node_name, vehicle_id):
        node = self.nodes[node_name]
        cur_speed = self.phase_map.get_speed(node.speed_id, action)
        speed = self.sim.vehicle.getSpeed(vehicle_id)
        for a in cur_speed:
            if a == 'k':
                speed_adv = speed
            elif a == 'G':
                speed_adv = speed
            elif a == 'g':
                speed_adv = speed
            elif a == 'r':
                speed_adv = speed
            elif a == 'y':
                speed_adv = speed
            elif a == 'a':
                speed_adv = self.sim.vehicle.getAllowedSpeed(vehicle_id) - 0.01
            elif a == 'l':
                speed_adv = speed
            elif a == 'm':
                speed_adv = speed
            elif a == 'n':
                speed_adv = speed
            elif a == 'o':
                speed_adv = speed
            elif a == 'p':
                speed_adv = speed
            elif a == 'q':
                speed_adv = speed
            elif a == 'z':
                speed_adv = speed
            elif a == 'd':
                speed_adv = speed
            if speed_adv < 0:
                speed_adv = 0.01
            if speed_adv == 0:
                speed_adv = 0.01
            # return speed_adv
            # speed_adv = cur_speed

        return speed_adv

    def _get_node_phase_id(self, node_name):
        raise NotImplementedError()

    def _get_node_signal_id(self, node_name):
        raise NotImplementedError()

    def _get_node_speed_id(self, node_name):
        raise NotImplementedError()

    def _get_node_state_num(self, node):
        # assert len(node.lanes_in) == self.phase_map.get_lane_num(node.phase_id)
        return len(node.ilds_in)

    def _get_state(self):
        state = []
        self._measure_state_step()

        for node_name in self.node_names:
            node = self.nodes[node_name]
            if self.agent == 'greedy':
                state.append(node.wave_state)
            elif self.agent == 'a2c':
                if 'wait' in self.state_names:
                    state.append(np.concatenate([node.wave_state, node.wait_state]))
                else:
                    state.append(node.wave_state)
            else:
                cur_state = [node.wave_state]
                for nnode_name in node.neighbor:
                    if self.agent != 'ma2c':
                        cur_state.append(self.nodes[nnode_name].wave_state)
                    else:
                        cur_state.append(self.nodes[nnode_name].wave_state * self.coop_gamma)
                if 'wait' in self.state_names:
                    cur_state.append(node.wait_state)
                if self.agent == 'ma2c':
                    for nnode_name in node.neighbor:
                        cur_state.append(self.nodes[nnode_name].fingerprint)
                state.append(np.concatenate(cur_state))

        if self.agent == 'a2c':
            state = np.concatenate(state)

        # # clean up the state and fingerprint measurements
        # for node in self.node_names:
        #     self.nodes[node].state = np.zeros(self.nodes[node].num_state)
        #     self.nodes[node].fingerprint = np.zeros(self.nodes[node].num_fingerprint)
        return state

    def _init_nodes(self):
        nodes = {}
        for node_name in self.sim.trafficlight.getIDList():
            if node_name in self.neighbor_map:
                neighbor = self.neighbor_map[node_name]
            else:
                logging.info('node %s can not be found!' % node_name)
                neighbor = []
            nodes[node_name] = Node(node_name,
                                    neighbor=neighbor,
                                    control=True)
            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)
            nodes[node_name].lanes_in = lanes_in
            # edges_in = []
            ilds_in = []
            for lane_name in lanes_in:
                ild_name = lane_name
                if ild_name not in ilds_in:
                    ilds_in.append(ild_name)

            # nodes[node_name].edges_in = edges_in
            nodes[node_name].ilds_in = ilds_in
        self.nodes = nodes
        self.node_names = sorted(list(nodes.keys()))
        s = 'Env: init %d node information:\n' % len(self.node_names)
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\tneigbor: %r\n' % node.neighbor
            # s += '\tlanes_in: %r\n' % node.lanes_in
            s += '\tilds_in: %r\n' % node.ilds_in
            # s += '\tedges_in: %r\n' % node.edges_in
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        self.n_a_ls = []
        self.n_a_lsp = []
        self.n_a_lss = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_id = self._get_node_phase_id(node_name)
            signal_id = self._get_node_signal_id(node_name)
            speed_id = self._get_node_speed_id(node_name)
            node.phase_id = phase_id
            node.signal_id = signal_id
            node.speed_id = speed_id
            node.n_a = self.phase_map.get_phase_num(phase_id)
            node.n_a_p = self.phase_map.get_signal_num(signal_id)
            node.n_a_s = self.phase_map.get_speed_num(speed_id)
            self.n_a_ls.append(node.n_a)
            self.n_a_lsp.append(node.n_a_p)
            self.n_a_lss.append(node.n_a_s)
        self.n_a = np.prod(np.array(self.n_a_ls))
        self.n_a_p = np.prod(np.array(self.n_a_lsp))
        self.n_a_s = np.prod(np.array(self.n_a_lss))

    def _init_map(self):
        self.neighbor_map = None
        self.phase_map = None
        self.state_names = None
        raise NotImplementedError()

    def _init_policy(self):
        policy = []
        for node_name in self.node_names:
            phase_num = self.nodes[node_name].n_a
            p = 1. / phase_num
            policy.append(np.array([p] * phase_num ))
        return policy

    def _init_sim(self, seed, gui=False):
        sumocfg_file = self._init_sim_config(seed)
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        if self.name != 'real_net':
            command += ['--time-to-teleport', '600']
        else:
            command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        if self.is_record:
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        subprocess.Popen(command)
        # wait 2s to establish the traci server
        time.sleep(2)
        self.sim = traci.connect(port=self.port)

        self.bus_stop_ids = self.sim.busstop.getIDList()

    def _init_sim_config(self):
        raise NotImplementedError()

    def _init_sim_traffic(self):
        return

    def _init_state_space(self):
        self._reset_state()
        self.n_s_ls = []
        self.n_w_ls = []
        self.n_f_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            num_wave = node.num_state
            num_fingerprint = 0
            for nnode_name in node.neighbor:
                if self.agent not in ['a2c', 'greedy']:
                    num_wave += self.nodes[nnode_name].num_state
                if self.agent == 'ma2c':
                    num_fingerprint += self.nodes[nnode_name].num_fingerprint
            num_wait = 0 if 'wait' not in self.state_names else node.num_state
            self.n_s_ls.append(num_wave + num_wait + num_fingerprint)
            self.n_f_ls.append(num_fingerprint)
            self.n_w_ls.append(num_wait)
        self.n_s = np.sum(np.array(self.n_s_ls))

    def _measure_reward_step(self):
        rewards = []
        first_twos = []
        last_ones = []

        queuess = []
        waitss = []
        CV_speedss = []
        HV_speedss = []
        deviationss = []
        for node_name in self.node_names:
            queues = []
            waits = []
            CV_speeds = []
            HV_speeds = []
            CV_ids = []
            HV_ids = []
            deviations = []
            for ild in self.nodes[node_name].ilds_in:
                if self.obj in ['queue', 'hybrid']:
                    if self.name == 'real_net':
                        cur_queue = min(10, self.sim.lane.getLastStepHaltingNumber(ild))
                    else:
                        cur_queue = self.sim.lanearea.getLastStepHaltingNumber(ild)
                    queues.append(cur_queue)
                if self.obj in ['wait', 'hybrid']:
                    max_pos = 0
                    car_wait = 0
                    if self.name == 'real_net':
                        cur_cars = self.sim.lane.getLastStepVehicleIDs(ild)
                    else:
                        cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
                    for vid in cur_cars:
                        car_pos = self.sim.vehicle.getLanePosition(vid)
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.sim.vehicle.getWaitingTime(vid)
                    waits.append(car_wait)
                if self.obj in ['speed', 'hybrid']:
                    car_ids = self.sim.lane.getLastStepVehicleIDs(ild)
                    for car_id in car_ids:
                        car_type = self.sim.vehicle.getTypeID(car_id)
                        if car_type == 'CV':
                            CV_ids.append(car_id)
                        elif car_type == 'HV':
                            HV_ids.append(car_id)
                    num_tot_CV = len(CV_ids)
                    if num_tot_CV > 0:
                        for CV_id in CV_ids:
                            CV_speed = self.sim.vehicle.getSpeed(CV_id)
                            CV_speeds.append(CV_speed)

                    num_tot_HV = len(HV_ids)
                    if num_tot_HV > 0:
                        for HV_id in HV_ids:
                            HV_speed = self.sim.vehicle.getSpeed(HV_id)
                            HV_speeds.append(HV_speed)

            if self.obj in ['deviation', 'hybrid']:
                for bus_stop_id in self.bus_stop_ids:
                    if bus_stop_id.startswith(node_name):
                        buss_at_stop = self.sim.busstop.getVehicleIDs(bus_stop_id)
                        if len(buss_at_stop):
                            for bus_at_stop in buss_at_stop:
                                cur_deviation = abs(int(bus_at_stop.split('_')[-1]) * 1200
                                                    + self.baselines[bus_stop_id] - self.sim.simulation.getTime())
                                deviations.append(cur_deviation)

            queue = np.sum(np.array(queues)) if len(queues) else 0
            wait = np.sum(np.array(waits)) if len(waits) else 0
            CV_speed = np.sum(np.array(CV_speeds) if len(CV_speeds) else 0)
            HV_speed = np.sum(np.array(HV_speeds) if len(HV_speeds) else 0)
            deviation = np.sum(np.array(deviations)) if len(deviations) else 0

            if self.obj == 'queue':
                reward = - queue
            elif self.obj == 'wait':
                reward = - wait
            # elif self.obj == 'speed':
            #     reward = -speed
            else:
                # reward =
                pass
            rewards.append(reward)
            first_twos.append(first_two)
            last_ones.append(last_one)

            queuess.append(queue)
            waitss.append(wait)
            CV_speedss.append(CV_speed)
            HV_speedss.append(HV_speed)
            deviationss.append(deviation)

        return np.array(rewards), np.array(first_twos), np.array(last_ones), np.array(queuess), np.array(waitss), np.array(CV_speedss), np.array(HV_speedss), np.array(deviationss)

    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names:
                if state_name == 'wave':
                    cur_state = []
                    for ild in node.ilds_in:
                        if self.name == 'real_net':
                            cur_wave = self.sim.lane.getLastStepVehicleNumber(ild)
                        else:
                            cur_wave = self.sim.lanearea.getLastStepVehicleNumber(ild)
                        cur_state.append(cur_wave)
                    cur_state = np.array(cur_state)
                else:
                    cur_state = []
                    for ild in node.ilds_in:
                        max_pos = 0
                        car_wait = 0
                        if self.name == 'real_net':
                            cur_cars = self.sim.lane.getLastStepVehicleIDs(ild)
                        else:
                            cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
                        for vid in cur_cars:
                            car_pos = self.sim.vehicle.getLanePosition(vid)
                            if car_pos > max_pos:
                                max_pos = car_pos
                                car_wait = self.sim.vehicle.getWaitingTime(vid)
                        cur_state.append(car_wait)
                    cur_state = np.array(cur_state)
                if self.record_stats:
                    self.state_stat[state_name] += list(cur_state)
                norm_cur_state = self._norm_clip_state(cur_state,
                                                       self.norms[state_name],
                                                       self.clips[state_name])
                if state_name == 'wave':
                    node.wave_state = norm_cur_state
                else:
                    node.wait_state = norm_cur_state

    def _measure_traffic_step(self):
        cars = self.sim.vehicle.getIDList()
        num_tot_car = len(cars)
        num_in_car = self.sim.simulation.getDepartedNumber()
        num_out_car = self.sim.simulation.getArrivedNumber()
        if num_tot_car > 0:
            avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars])
            avg_speed = np.mean([self.sim.vehicle.getSpeed(car) for car in cars])
        else:
            avg_speed = 0
            avg_waiting_time = 0

        queues = []
        for node_name in self.node_names:
            for ild in self.nodes[node_name].ilds_in:
                queues.append(self.sim.lane.getLastStepHaltingNumber(ild))
        avg_queue = np.mean(np.array(queues))
        std_queue = np.std(np.array(queues))
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}
        self.traffic_data.append(cur_traffic)

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = x / norm
        return x if clip < 0 else np.clip(x, 0, clip)

    def _reset_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            node.prev_action = 0
            node.num_fingerprint = node.n_a
            node.num_state = self._get_node_state_num(node)
            # node.waves = np.zeros(node.num_state)
            # node.waits = np.zeros(node.num_state)

    def _set_phase(self, action_p, phase_type, phase_duration):
        for node_name, a in zip(self.node_names, list(action_p)):
            phase = self._get_node_phase(a, node_name, phase_type)
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
            self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)

    def _set_vehicle(self, node_name, a):

        CV_ids = []
        HV_ids = []
        speed_advs = []

        edges_id = self.nodes[node_name].ilds_in
        for edge_id in edges_id:

            vehicle_ids = self.sim.lane.getLastStepVehicleIDs(edge_id)

            for v_id in vehicle_ids:

                v_type = self.sim.vehicle.getTypeID(v_id)

                if v_type == 'CV':
                    CV_ids.append(v_id)
                # elif v_type == 'HV':
                #     HV_ids.append(v_id)

        for CV_id in CV_ids:
            if random.uniform(0, 1) < 0.8:
                speed_adv = self._get_vehicle_speed(a, node_name, CV_id)

                self.sim.vehicle.slowDown(CV_id, speed_adv, 0.00001)

    def _set_speed(self, action_s):
        for node_name, a in zip(self.node_names, list(action_s)):

            self._set_vehicle(node_name, a)

    def _simulate(self, num_step):
        for _ in range(num_step):
            self.sim.simulationStep()
            # self._measure_state_step()
            # reward += self._measure_reward_step()
            self.cur_sec += 1
            if self.is_record:
                # self._debug_traffic_step()
                self._measure_traffic_step()
        # return reward

    def _transfer_action(self, action_p, action_s):
        '''Transfer global action to a list of local actions'''
        phase_nums = []
        speed_nums = []
        signal_nums = []
        for node in self.control_node_names:
            phase_nums.append(self.nodes[node].phase_num)
            speed_nums.append(self.nodes[node].speed_num)
            signal_nums.append(self.nodes[node].signal_num)
        action_lsp = []
        for i in range(len(signal_nums) - 1):
            action_p, cur_action_p = divmod(action_p, signal_nums[i])
            action_lsp.append(cur_action_p)
        action_lsp.append(action_p)
        action_lss = []
        for i in range(len(speed_nums) - 1):
            action_s, cur_action_s = divmod(action_s, speed_nums[i])
            action_lss.append(cur_action_s)
        action_lss.append(action_s)
        return action_lsp, action_lss

    def _update_waits(self, action):
        for node_name, a in zip(self.node_names, action):
            red_lanes = set()
            node = self.nodes[node_name]
            for i in self.phase_map.get_red_lanes(node.phase_id, a):
                red_lanes.add(node.lanes_in[i])
            for i in range(len(node.waits)):
                lane = node.ilds_in[i]
                if lane in red_lanes:
                    node.waits[i] += self.control_interval_sec
                else:
                    node.waits[i] = 0

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))
        tree = ET.ElementTree(file=trip_file)
        for child in tree.getroot():
            cur_trip = child.attrib
            cur_dict = {}
            cur_dict['episode'] = self.cur_episode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitingCount']
            cur_dict['wait_sec'] = cur_trip['waitingTime']
            self.trip_data.append(cur_dict)
        # delete the current xml
        # cmd = 'rm ' + trip_file
        # subprocess.check_call(cmd, shell=True)
        os.remove(trip_file)
        print('File Deleted Successfully')

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.record_stats = record_stats
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []
        if self.record_stats:
            self.state_stat = {}
            for state_name in self.state_names:
                self.state_stat[state_name] = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.agent)))

    def reset(self, gui=False, test_ind=0):
        self._reset_state()
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        # self._init_sim(gui=True)
        self._init_sim(seed, gui=gui)
        self.cur_sec = 0
        self.cur_episode += 1
        if self.agent == 'ma2c':
            self.update_fingerprint(self._init_policy())
        self._init_sim_traffic()
        self.seed += 1
        return self._get_state()

    def terminate(self):
        self.sim.close()

        self.pre_deviation = 0

        self.n_episode += 1
        if self.n_episode == 0:
            print('n_episode:', self.n_episode, 'time:', time.asctime(),
                  'cost_time:', (time.time() - self.pre_time) / 60, 'total_time:', (time.time() - self.start_time) / 60,
                  )
            logging.info('n_episode: %r' % self.n_episode)
        else:
            print('n_episode:', self.n_episode, 'time:', time.asctime(),
                  'cost_time:', (time.time() - self.pre_time) / 60, 'total_time:', (time.time() - self.start_time) / 60,
                  'rest_time:', (time.time() - self.start_time) / 60 / self.n_episode * (1389 - self.n_episode)
                  )
            logging.info('n_episode: %r' % self.n_episode)
        self.pre_time = time.time()


    def step(self, action_p, action_s):
        if self.agent == 'a2c':
            action_p, action_s = self._transfer_action(action_p, action_s)
        # self._update_waits(action)
        self._set_phase(action_p, 'yellow', self.yellow_interval_sec)
        self._simulate(self.yellow_interval_sec)
        rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec
        self._set_phase(action_p, 'green', rest_interval_sec)
        self._simulate(rest_interval_sec)
        self._set_speed(action_s)
        # self._simulate(rest_interval_sec)

        state = self._get_state()
        reward, first_two, last_one, queue, wait, CV_speed, HV_speed, deviation = self._measure_reward_step()
        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True

        global_reward = np.sum(reward)
        global_first_two = np.sum(first_two)
        global_last_one = np.sum(last_one)

        # logging.info('local reward: %r, global_reward: %r, queue: %r, wait: %r, CV_speed: %r , HV_speed: %r '%
        #              (reward, global_reward, queue, wait, CV_speed, HV_speed))
        if self.is_record:
            action_r_p = ','.join(['%d' % a for a in action_p])
            action_r_s = ','.join(['%d' % a for a in action_s])
            cur_control = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'step': self.cur_sec / self.control_interval_sec,
                           'action_p': action_r_p,
                           'action_s': action_r_s,
                           'reward': global_reward,
                           'queue': queue,
                           'sum(queue)': sum(queue),
                           'wait': wait,
                           'sum(wait)': sum(wait),
                           'CV_speed': CV_speed,
                           'sum(CV_speed)': sum(CV_speed),
                           'HV_speed': HV_speed,
                           'sum(HV_speed)': sum(HV_speed),
                           'deviation': deviation,
                           'sum(deviation)': sum(deviation),
                           }
            self.control_data.append(cur_control)

        if not self.train_mode:
            return state, first_two, done, global_reward, global_first_two, global_last_one
        if self.agent in ['a2c', 'greedy']:
            first_two = global_first_two
        elif self.agent != 'ma2c':
            new_first_two = [global_first_two] * len(first_two)
            first_two = np.array(new_first_two)
            if self.name == 'real_net':
                first_two = first_two / (len(self.node_names) * REALNET_REWARD_NORM)
        else:
            new_first_two = []
            for node_name, r in zip(self.node_names, first_two):
                cur_first_two = r
                for nnode_name in self.nodes[node_name].neighbor:
                    i = self.node_names.index(nnode_name)
                    cur_first_two += self.coop_gamma * first_two[i]
                if self.name != 'real_net':
                    new_first_two.append(cur_first_two)
                else:
                    n_node = 1 + len(self.nodes[node_name].neighbor)
                    new_first_two.append(cur_first_two / (n_node * REALNET_REWARD_NORM))
            first_two = np.array(new_first_two)
        return state, first_two, done, global_reward, global_first_two, global_last_one

    def update_fingerprint(self, policy):
        for node_name, pi in zip(self.node_names, policy):
            # self.nodes[node_name].fingerprint = np.array(pi)[:-1]
            self.nodes[node_name].fingerprint = np.array(pi)
