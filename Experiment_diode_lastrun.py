#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.2a1),
    on September 28, 2026, at 15:46
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, iohub
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from init_code
###############################################################################

#---ANT related---#
import pyxid2
import threading
import signal
import pandas as pd
import random
import json
import os

def exit_after(s):
    '''
    function decorator to raise KeyboardInterrupt exception
    if function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, signal.raise_signal, args=[signal.SIGINT])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@exit_after(1)  # exit if function takes longer than 1 seconds
def _get_xid_devices():
    return pyxid2.get_xid_devices()


def get_xid_devices():
    print("Getting a list of all attached XID devices...")
    attempt_count = 0
    while attempt_count >= 0:
        attempt_count += 1
        print('     Attempt:', attempt_count)
        attempt_count *= -1  # try to exit the while loop
        try:
            devices = _get_xid_devices()
        except KeyboardInterrupt:
            attempt_count *= -1  # get back in the while loop
    return devices


devices = get_xid_devices()

if devices:
    dev = devices[0]
    print("Found device:", dev)
    assert dev.device_name == 'Cedrus C-POD', "Incorrect C-POD detected." #need to change to trigger box
    dev.set_pulse_duration(5)  # set pulse duration to 5ms

    # Start EEG recording
    print("Sending trigger code 126 to start EEG recording...")
    dev.activate_line(bitmask=126)  # trigger 126 will start EEG
    print("Waiting 10 seconds for the EEG recording to start...\n")
    core.wait(10)  # wait 10s for the EEG system to start recording

    # Marching lights test
    print("C-POD<->eego 7-bit trigger lines test...")
    for line in range(1, 8):  # raise lines 1-7 one at a time
        print("  raising line {} (bitmask {})".format(line, 2 ** (line-1)))
        dev.activate_line(lines=line)
        core.wait(0.5)  # wait 500ms between two consecutive triggers
    dev.con.set_digio_lines_to_mask(0)  # XidDevice.clear_all_lines()
    print("EEG system is now ready for the experiment to start.\n")

else:
    # Dummy XidDevice for code components to run without C-POD connected
    class dummyXidDevice(object):
        def __init__(self):
            pass
        def activate_line(self, lines=None, bitmask=None):
            pass


    print("WARNING: No C-POD connected for this session! "
          "You must start/stop EEG recording manually!\n")
    dev = dummyXidDevice()
#------#


###############################################################################
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.2a1'
expName = 'Experiment_diode'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'start_block': '1',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\purdo\\Documents\\Wagner Lab\\FSA\\Experiment_diode_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # log the filename of last_app_load.log
    print('target_last_app_load_log_file: ' + filename + '_last_app_load.log')
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup eyetracking
    ioConfig['eyetracker.eyelink.EyeTracker'] = {
        'name': 'tracker',
        'model_name': 'EYELINK 1000 DESKTOP',
        'simulation_mode': False,
        'network_settings': '100.1.1.1',
        'default_native_data_file_name': 'EXPFILE',
        'runtime_settings': {
            'sampling_rate': 1000.0,
            'track_eyes': 'LEFT_EYE',
            'sample_filtering': {
                'FILTER_FILE': 'FILTER_LEVEL_2',
                'FILTER_ONLINE': 'FILTER_LEVEL_OFF',
            },
            'vog_settings': {
                'pupil_measure_types': 'PUPIL_AREA',
                'tracking_mode': 'PUPIL_CR_TRACKING',
                'pupil_center_algorithm': 'ELLIPSE_FIT',
            }
        }
    }
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    deviceManager.devices['eyetracker'] = ioServer.getDevice('tracker')
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('progress') is None:
        # initialise progress
        progress = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='progress',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('between_block_resp') is None:
        # initialise between_block_resp
        between_block_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='between_block_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "init" ---
    # Run 'Begin Experiment' code from init_code
    ##########################################
    EEG_cue_onset=10
    EEG_stim_onset=11
    
    ##########################################
    
    # --- Initialize components for Routine "instructions" ---
    progress = keyboard.Keyboard(deviceName='progress')
    instruction = visual.TextStim(win=win, name='instruction',
        text='Please wait for the experimenter to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from code_5
    # Create 60 unique jitters and shuffle them
    jitter_list = [round(1.0 + i * (1/60), 6) for i in range(61)]
    np.random.shuffle(jitter_list)
    jitter_index = 0
    
    # Check jitters
    print(f"Jitter list length: {len(jitter_list)}")
    print(f"Min: {min(jitter_list)}, Max: {max(jitter_list)}")
    print(f"Step size: {round(jitter_list[1] - jitter_list[0], 6)}")
    
    import json
    import random
    
    # Builds a 60-trial cueDir sequence with no more than `max_consecutive`
    # same-side trials in a row. Used by code_9 to build each block's actual
    # trial order (fed into the 'trials' loop below), so this constraint is
    # real and not just generated-and-discarded.
    def generate_constrained_sequence(n_valid_left, n_valid_right,
                                       max_consecutive=5, max_attempts=10000):
        trial_pool = []
        for _ in range(n_valid_left):
            trial_pool.append({'cueDir': 'left', 'valid': 1, 'trialType': 'valid_left'})
        for _ in range(n_valid_right):
            trial_pool.append({'cueDir': 'right', 'valid': 1, 'trialType': 'valid_right'})
    
        for attempt in range(max_attempts):
            random.shuffle(trial_pool)
            valid_sequence = True
            consecutive = 1
            for i in range(1, len(trial_pool)):
                if trial_pool[i]['cueDir'] == trial_pool[i-1]['cueDir']:
                    consecutive += 1
                    if consecutive > max_consecutive:
                        valid_sequence = False
                        break
                else:
                    consecutive = 1
            if valid_sequence:
                return list(trial_pool)
    
        print("Warning: could not find valid sequence, using last attempt")
        return list(trial_pool)
    
    def generate_constrained_tilt(n_total, max_consecutive=4, max_attempts=10000):
        tilt_pool = [1] * (n_total // 2) + [-1] * (n_total // 2)
        for attempt in range(max_attempts):
            random.shuffle(tilt_pool)
            valid = True
            consecutive = 1
            for i in range(1, len(tilt_pool)):
                if tilt_pool[i] == tilt_pool[i-1]:
                    consecutive += 1
                    if consecutive > max_consecutive:
                        valid = False
                        break
                else:
                    consecutive = 1
            if valid:
                return list(tilt_pool)
        print("Warning: could not find valid tilt sequence, using last attempt")
        return list(tilt_pool)
    
    n_valid_left = 30
    n_valid_right = 30
    
    threshold_file = f"data/threshold_{expInfo['participant']}.json"
    if os.path.exists(threshold_file):
        with open(threshold_file, 'r') as f:
            threshold_data = json.load(f)
        level = threshold_data['threshold_angle']
        print(f"Loaded threshold angle for participant {expInfo['participant']} from {threshold_file}: {level:.2f} degrees")
    else:
        level = 15.0
        print(f"Warning: no threshold file found at {threshold_file}, using default 15 degrees")
    thisExp.addData('threshold_angle_used', level)
    
    # Block tracking - start_block input specifies which block the participant starts on
    current_block = int(expInfo['start_block'])
    target_blocks = 8  # soft target - experiment continues past this if needed
    
    # Cumulative accuracy tracking across all blocks
    expInfo['total_correct'] = 0
    expInfo['total_trials'] = 0
    expInfo['total_noresp'] = 0
    expInfo['completed_blocks'] = 0
    
    # --- Initialize components for Routine "eye_start" ---
    etRecord_start = hardware.eyetracker.EyetrackerControl(
        tracker=eyetracker,
        actionType='Start Only'
    )
    
    # --- Initialize components for Routine "block_control" ---
    
    # --- Initialize components for Routine "fixation" ---
    fixcross = visual.TextStim(win=win, name='fixcross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "cue" ---
    cue_circle = visual.ShapeStim(
        win=win, name='cue_circle',units='height', 
        size=(0.1, 0.1), vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=5.0,
        colorSpace='rgb', lineColor='white', fillColor=[0,0,0],
        opacity=None, depth=-1.0, interpolate=True)
    square_upper_left = visual.Rect(
        win=win, name='square_upper_left',units='pix', 
        width=(50,50)[0], height=(50,50)[1],
        ori=0.0, pos=(-win.size[0]/2,win.size[1]/2), draggable=False, anchor='top-left',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-2.0, interpolate=True)
    square_bottom_right = visual.Rect(
        win=win, name='square_bottom_right',units='pix', 
        width=(50, 50)[0], height=(50, 50)[1],
        ori=0.0, pos=(win.size[0]/2, -win.size[1]/2), draggable=False, anchor='bottom-right',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "cue_to_target" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "gabor_patch" ---
    gabor = visual.GratingStim(
        win=win, name='gabor',units='height', 
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=[0,0], draggable=False, size=(0.1, 0.1), sf=5.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=0.1, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-1.0)
    square_upper_left2 = visual.Rect(
        win=win, name='square_upper_left2',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=(-1, -1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    square_bottom_right2 = visual.Rect(
        win=win, name='square_bottom_right2',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=(1, 1), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "response" ---
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "eye_stop" ---
    etRecord_stop = hardware.eyetracker.EyetrackerControl(
        tracker=eyetracker,
        actionType='Stop Only'
    )
    
    # --- Initialize components for Routine "between_block" ---
    end_block_text = visual.TextStim(win=win, name='end_block_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    between_block_resp = keyboard.Keyboard(deviceName='between_block_resp')
    
    # --- Initialize components for Routine "endroutine" ---
    text = visual.TextStim(win=win, name='text',
        text=' ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "init" ---
    # create an object to store info about Routine init
    init = data.Routine(
        name='init',
        components=[],
    )
    init.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for init
    init.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    init.tStart = globalClock.getTime(format='float')
    init.status = STARTED
    thisExp.addData('init.started', init.tStart)
    init.maxDuration = None
    # keep track of which components have finished
    initComponents = init.components
    for thisComponent in init.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "init" ---
    init.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            init.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in init.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "init" ---
    for thisComponent in init.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for init
    init.tStop = globalClock.getTime(format='float')
    init.tStopRefresh = tThisFlipGlobal
    thisExp.addData('init.stopped', init.tStop)
    thisExp.nextEntry()
    # the Routine "init" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # define target for eye_calibration
    eye_calibrationTarget = visual.TargetStim(win, 
        name='eye_calibrationTarget',
        radius=0.015, fillColor='white', borderColor='green', lineWidth=2.0,
        innerRadius=0.005, innerFillColor='black', innerBorderColor='black', innerLineWidth=2.0,
        colorSpace='rgb', units=None
    )
    # define parameters for eye_calibration
    eye_calibration = hardware.eyetracker.EyetrackerCalibration(win, 
        eyetracker, eye_calibrationTarget,
        units=None, colorSpace='rgb',
        progressMode='time', targetDur=1.5, expandScale=1.5,
        targetLayout='NINE_POINTS', randomisePos=True, textColor='white',
        movementAnimation=True, targetDelay=1.0
    )
    # run calibration
    eye_calibration.run()
    # clear any keypresses from during eye_calibration so they don't interfere with the experiment
    defaultKeyboard.clearEvents()
    thisExp.nextEntry()
    # the Routine "eye_calibration" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[progress, instruction],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for progress
    progress.keys = []
    progress.rt = []
    _progress_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *progress* updates
        waitOnFlip = False
        
        # if progress is starting this frame...
        if progress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            progress.frameNStart = frameN  # exact frame index
            progress.tStart = t  # local t and not account for scr refresh
            progress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(progress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'progress.started')
            # update status
            progress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(progress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(progress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if progress.status == STARTED and not waitOnFlip:
            theseKeys = progress.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _progress_allKeys.extend(theseKeys)
            if len(_progress_allKeys):
                progress.keys = _progress_allKeys[-1].name  # just the last key pressed
                progress.rt = _progress_allKeys[-1].rt
                progress.duration = _progress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instruction* updates
        
        # if instruction is starting this frame...
        if instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction.frameNStart = frameN  # exact frame index
            instruction.tStart = t  # local t and not account for scr refresh
            instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction.started')
            # update status
            instruction.status = STARTED
            instruction.setAutoDraw(True)
        
        # if instruction is active this frame...
        if instruction.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if progress.keys in ['', [], None]:  # No response was made
        progress.keys = None
    thisExp.addData('progress.keys',progress.keys)
    if progress.keys != None:  # we had a response
        thisExp.addData('progress.rt', progress.rt)
        thisExp.addData('progress.duration', progress.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=999.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "eye_start" ---
        # create an object to store info about Routine eye_start
        eye_start = data.Routine(
            name='eye_start',
            components=[etRecord_start],
        )
        eye_start.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_eye_start_trigger
        win.callOnFlip(eyetracker.sendMessage, f"BLOCK_{current_block}_REC_START")
        thisExp.addData('eyetracker_rec_start_block', current_block)
        # store start times for eye_start
        eye_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        eye_start.tStart = globalClock.getTime(format='float')
        eye_start.status = STARTED
        thisExp.addData('eye_start.started', eye_start.tStart)
        eye_start.maxDuration = None
        # keep track of which components have finished
        eye_startComponents = eye_start.components
        for thisComponent in eye_start.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "eye_start" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        eye_start.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *etRecord_start* updates
            
            # if etRecord_start is starting this frame...
            if etRecord_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                etRecord_start.frameNStart = frameN  # exact frame index
                etRecord_start.tStart = t  # local t and not account for scr refresh
                etRecord_start.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(etRecord_start, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('etRecord_start.started', t)
                # update status
                etRecord_start.status = STARTED
                etRecord_start.start()
            if etRecord_start.status == STARTED:
                etRecord_start.tStop = t  # not accounting for scr refresh
                etRecord_start.tStopRefresh = tThisFlipGlobal  # on global time
                etRecord_start.frameNStop = frameN  # exact frame index
                etRecord_start.status = FINISHED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                eye_start.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in eye_start.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "eye_start" ---
        for thisComponent in eye_start.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for eye_start
        eye_start.tStop = globalClock.getTime(format='float')
        eye_start.tStopRefresh = tThisFlipGlobal
        thisExp.addData('eye_start.stopped', eye_start.tStop)
        # the Routine "eye_start" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "block_control" ---
        # create an object to store info about Routine block_control
        block_control = data.Routine(
            name='block_control',
            components=[],
        )
        block_control.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_9
        # Regenerate jitter list fresh each block with new random order
        jitter_list = [round(1.0 + i * (1/60), 6) for i in range(61)]
        np.random.shuffle(jitter_list)
        jitter_index = 0
        
        # Build this block's actual trial order (max 4 consecutive same-side cues),
        # fed directly into the 'trials' loop via its conditionsFile = $block_trial_list_file
        block_trial_list = generate_constrained_sequence(n_valid_left, n_valid_right, max_consecutive=4)
        tilt_sequence = generate_constrained_tilt(len(block_trial_list), max_consecutive=4)
        for i, trial in enumerate(block_trial_list):
            trial['tiltDir'] = tilt_sequence[i]
            trial['corr_response'] = '2' if trial['tiltDir'] == 1 else '1'
        
        # data.importConditions() needs an actual file, not a Python list, so write
        # this block's constrained trial order to a small CSV and point the 'trials'
        # loop at that file (conditionsFile = $block_trial_list_file)
        import csv
        block_trial_list_file = f"data/_block_trial_list_{expInfo['participant']}.csv"
        with open(block_trial_list_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['cueDir', 'valid', 'trialType', 'corr_response', 'tiltDir'])
            writer.writeheader()
            writer.writerows(block_trial_list)
        
        # Reset per-block counters
        expInfo[f'block_{current_block}_correct'] = 0
        expInfo[f'block_{current_block}_total'] = 0
        expInfo[f'block_{current_block}_noresp'] = 0
        
        print(f"\n{'='*50}")
        print(f"Starting Block {current_block}")
        print(f"{'='*50}")
        # store start times for block_control
        block_control.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_control.tStart = globalClock.getTime(format='float')
        block_control.status = STARTED
        thisExp.addData('block_control.started', block_control.tStart)
        block_control.maxDuration = None
        # keep track of which components have finished
        block_controlComponents = block_control.components
        for thisComponent in block_control.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_control" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        block_control.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_control.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_control.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_control" ---
        for thisComponent in block_control.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_control
        block_control.tStop = globalClock.getTime(format='float')
        block_control.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_control.stopped', block_control.tStop)
        # the Routine "block_control" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(block_trial_list_file), 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "fixation" ---
            # create an object to store info about Routine fixation
            fixation = data.Routine(
                name='fixation',
                components=[fixcross],
            )
            fixation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_8
            # In your fixation Begin Routine
            win.callOnFlip(eyetracker.sendMessage, 
                           f"TRIAL_START {trials.thisN} cueDir={cueDir}")
            thisExp.addData('block_number', current_block)
            # store start times for fixation
            fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixation.tStart = globalClock.getTime(format='float')
            fixation.status = STARTED
            thisExp.addData('fixation.started', fixation.tStart)
            fixation.maxDuration = None
            # keep track of which components have finished
            fixationComponents = fixation.components
            for thisComponent in fixation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            fixation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixcross* updates
                
                # if fixcross is starting this frame...
                if fixcross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixcross.frameNStart = frameN  # exact frame index
                    fixcross.tStart = t  # local t and not account for scr refresh
                    fixcross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixcross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixcross.started')
                    # update status
                    fixcross.status = STARTED
                    fixcross.setAutoDraw(True)
                
                # if fixcross is active this frame...
                if fixcross.status == STARTED:
                    # update params
                    pass
                
                # if fixcross is stopping this frame...
                if fixcross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixcross.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        fixcross.tStop = t  # not accounting for scr refresh
                        fixcross.tStopRefresh = tThisFlipGlobal  # on global time
                        fixcross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixcross.stopped')
                        # update status
                        fixcross.status = FINISHED
                        fixcross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fixation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixation
            fixation.tStop = globalClock.getTime(format='float')
            fixation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixation.stopped', fixation.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if fixation.maxDurationReached:
                routineTimer.addTime(-fixation.maxDuration)
            elif fixation.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            
            # --- Prepare to start Routine "cue" ---
            # create an object to store info about Routine cue
            cue = data.Routine(
                name='cue',
                components=[cue_circle, square_upper_left, square_bottom_right],
            )
            cue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_2
            if cueDir == 'left':
                cuePos = (-0.5, 0)
            else:
                cuePos = (0.5, 0)
                
            #trigger status
            cue_trig=False
            
            cue_circle.setPos(cuePos)
            # store start times for cue
            cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            cue.tStart = globalClock.getTime(format='float')
            cue.status = STARTED
            thisExp.addData('cue.started', cue.tStart)
            cue.maxDuration = None
            # keep track of which components have finished
            cueComponents = cue.components
            for thisComponent in cue.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "cue" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            cue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.1:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code_2
                # In your cue Each Frame
                if cue_circle.status == STARTED and not cue_trig:
                    win.callOnFlip(dev.activate_line, bitmask=EEG_cue_onset)
                    win.callOnFlip(eyetracker.sendMessage,
                               f"CUE_ONSET dir={cueDir} trial={trials.thisN}")
                    cue_trig=True
                
                
                
                # *cue_circle* updates
                
                # if cue_circle is starting this frame...
                if cue_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue_circle.frameNStart = frameN  # exact frame index
                    cue_circle.tStart = t  # local t and not account for scr refresh
                    cue_circle.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue_circle, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_circle.started')
                    # update status
                    cue_circle.status = STARTED
                    cue_circle.setAutoDraw(True)
                
                # if cue_circle is active this frame...
                if cue_circle.status == STARTED:
                    # update params
                    pass
                
                # if cue_circle is stopping this frame...
                if cue_circle.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cue_circle.tStartRefresh + 0.1-frameTolerance:
                        # keep track of stop time/frame for later
                        cue_circle.tStop = t  # not accounting for scr refresh
                        cue_circle.tStopRefresh = tThisFlipGlobal  # on global time
                        cue_circle.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue_circle.stopped')
                        # update status
                        cue_circle.status = FINISHED
                        cue_circle.setAutoDraw(False)
                
                # *square_upper_left* updates
                
                # if square_upper_left is starting this frame...
                if square_upper_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    square_upper_left.frameNStart = frameN  # exact frame index
                    square_upper_left.tStart = t  # local t and not account for scr refresh
                    square_upper_left.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(square_upper_left, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'square_upper_left.started')
                    # update status
                    square_upper_left.status = STARTED
                    square_upper_left.setAutoDraw(True)
                
                # if square_upper_left is active this frame...
                if square_upper_left.status == STARTED:
                    # update params
                    pass
                
                # if square_upper_left is stopping this frame...
                if square_upper_left.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > square_upper_left.tStartRefresh + 0.1-frameTolerance:
                        # keep track of stop time/frame for later
                        square_upper_left.tStop = t  # not accounting for scr refresh
                        square_upper_left.tStopRefresh = tThisFlipGlobal  # on global time
                        square_upper_left.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'square_upper_left.stopped')
                        # update status
                        square_upper_left.status = FINISHED
                        square_upper_left.setAutoDraw(False)
                
                # *square_bottom_right* updates
                
                # if square_bottom_right is starting this frame...
                if square_bottom_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    square_bottom_right.frameNStart = frameN  # exact frame index
                    square_bottom_right.tStart = t  # local t and not account for scr refresh
                    square_bottom_right.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(square_bottom_right, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'square_bottom_right.started')
                    # update status
                    square_bottom_right.status = STARTED
                    square_bottom_right.setAutoDraw(True)
                
                # if square_bottom_right is active this frame...
                if square_bottom_right.status == STARTED:
                    # update params
                    pass
                
                # if square_bottom_right is stopping this frame...
                if square_bottom_right.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > square_bottom_right.tStartRefresh + 0.1-frameTolerance:
                        # keep track of stop time/frame for later
                        square_bottom_right.tStop = t  # not accounting for scr refresh
                        square_bottom_right.tStopRefresh = tThisFlipGlobal  # on global time
                        square_bottom_right.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'square_bottom_right.stopped')
                        # update status
                        square_bottom_right.status = FINISHED
                        square_bottom_right.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    cue.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in cue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "cue" ---
            for thisComponent in cue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for cue
            cue.tStop = globalClock.getTime(format='float')
            cue.tStopRefresh = tThisFlipGlobal
            thisExp.addData('cue.stopped', cue.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if cue.maxDurationReached:
                routineTimer.addTime(-cue.maxDuration)
            elif cue.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.100000)
            
            # --- Prepare to start Routine "cue_to_target" ---
            # create an object to store info about Routine cue_to_target
            cue_to_target = data.Routine(
                name='cue_to_target',
                components=[text_2],
            )
            cue_to_target.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_6
            jitter = jitter_list[jitter_index]
            jitter_index += 1
            text_2.setText(' ')
            # store start times for cue_to_target
            cue_to_target.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            cue_to_target.tStart = globalClock.getTime(format='float')
            cue_to_target.status = STARTED
            thisExp.addData('cue_to_target.started', cue_to_target.tStart)
            cue_to_target.maxDuration = None
            # keep track of which components have finished
            cue_to_targetComponents = cue_to_target.components
            for thisComponent in cue_to_target.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "cue_to_target" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            cue_to_target.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_2* updates
                
                # if text_2 is starting this frame...
                if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_2.frameNStart = frameN  # exact frame index
                    text_2.tStart = t  # local t and not account for scr refresh
                    text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.started')
                    # update status
                    text_2.status = STARTED
                    text_2.setAutoDraw(True)
                
                # if text_2 is active this frame...
                if text_2.status == STARTED:
                    # update params
                    pass
                
                # if text_2 is stopping this frame...
                if text_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_2.tStartRefresh + jitter-frameTolerance:
                        # keep track of stop time/frame for later
                        text_2.tStop = t  # not accounting for scr refresh
                        text_2.tStopRefresh = tThisFlipGlobal  # on global time
                        text_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_2.stopped')
                        # update status
                        text_2.status = FINISHED
                        text_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    cue_to_target.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in cue_to_target.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "cue_to_target" ---
            for thisComponent in cue_to_target.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for cue_to_target
            cue_to_target.tStop = globalClock.getTime(format='float')
            cue_to_target.tStopRefresh = tThisFlipGlobal
            thisExp.addData('cue_to_target.stopped', cue_to_target.tStop)
            # the Routine "cue_to_target" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "gabor_patch" ---
            # create an object to store info about Routine gabor_patch
            gabor_patch = data.Routine(
                name='gabor_patch',
                components=[gabor, square_upper_left2, square_bottom_right2],
            )
            gabor_patch.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            if valid == 1:
                stimLoc = cueDir
            else:
                stimLoc = 'right' if cueDir == 'left' else 'left'
            
            stimPos = (-0.5, 0) if stimLoc == 'left' else (0.5, 0)
            gabor_angle = level * tiltDir
            gabor.ori = gabor_angle
            gabor.setPos(stimPos)
            gabor.sf = 5
            
            # trigger sent status
            stim_trig=False
            
            gabor.setPos(stimPos)
            gabor.setOri(gabor_angle)
            # store start times for gabor_patch
            gabor_patch.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            gabor_patch.tStart = globalClock.getTime(format='float')
            gabor_patch.status = STARTED
            thisExp.addData('gabor_patch.started', gabor_patch.tStart)
            gabor_patch.maxDuration = None
            # keep track of which components have finished
            gabor_patchComponents = gabor_patch.components
            for thisComponent in gabor_patch.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "gabor_patch" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            gabor_patch.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.1:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code
                if gabor.status == STARTED and not stim_trig:
                    win.callOnFlip(dev.activate_line, bitmask=EEG_stim_onset)
                    win.callOnFlip(eyetracker.sendMessage,
                               f"GABOR_ONSET valid={valid} angle={gabor_angle:.2f}")
                    stim_trig=True
                
                
                # *gabor* updates
                
                # if gabor is starting this frame...
                if gabor.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    gabor.frameNStart = frameN  # exact frame index
                    gabor.tStart = t  # local t and not account for scr refresh
                    gabor.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(gabor, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'gabor.started')
                    # update status
                    gabor.status = STARTED
                    gabor.setAutoDraw(True)
                
                # if gabor is active this frame...
                if gabor.status == STARTED:
                    # update params
                    pass
                
                # if gabor is stopping this frame...
                if gabor.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > gabor.tStartRefresh + 0.1-frameTolerance:
                        # keep track of stop time/frame for later
                        gabor.tStop = t  # not accounting for scr refresh
                        gabor.tStopRefresh = tThisFlipGlobal  # on global time
                        gabor.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'gabor.stopped')
                        # update status
                        gabor.status = FINISHED
                        gabor.setAutoDraw(False)
                
                # *square_upper_left2* updates
                
                # if square_upper_left2 is starting this frame...
                if square_upper_left2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    square_upper_left2.frameNStart = frameN  # exact frame index
                    square_upper_left2.tStart = t  # local t and not account for scr refresh
                    square_upper_left2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(square_upper_left2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'square_upper_left2.started')
                    # update status
                    square_upper_left2.status = STARTED
                    square_upper_left2.setAutoDraw(True)
                
                # if square_upper_left2 is active this frame...
                if square_upper_left2.status == STARTED:
                    # update params
                    pass
                
                # if square_upper_left2 is stopping this frame...
                if square_upper_left2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > square_upper_left2.tStartRefresh + 0.1-frameTolerance:
                        # keep track of stop time/frame for later
                        square_upper_left2.tStop = t  # not accounting for scr refresh
                        square_upper_left2.tStopRefresh = tThisFlipGlobal  # on global time
                        square_upper_left2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'square_upper_left2.stopped')
                        # update status
                        square_upper_left2.status = FINISHED
                        square_upper_left2.setAutoDraw(False)
                
                # *square_bottom_right2* updates
                
                # if square_bottom_right2 is starting this frame...
                if square_bottom_right2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    square_bottom_right2.frameNStart = frameN  # exact frame index
                    square_bottom_right2.tStart = t  # local t and not account for scr refresh
                    square_bottom_right2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(square_bottom_right2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'square_bottom_right2.started')
                    # update status
                    square_bottom_right2.status = STARTED
                    square_bottom_right2.setAutoDraw(True)
                
                # if square_bottom_right2 is active this frame...
                if square_bottom_right2.status == STARTED:
                    # update params
                    pass
                
                # if square_bottom_right2 is stopping this frame...
                if square_bottom_right2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > square_bottom_right2.tStartRefresh + 0.1-frameTolerance:
                        # keep track of stop time/frame for later
                        square_bottom_right2.tStop = t  # not accounting for scr refresh
                        square_bottom_right2.tStopRefresh = tThisFlipGlobal  # on global time
                        square_bottom_right2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'square_bottom_right2.stopped')
                        # update status
                        square_bottom_right2.status = FINISHED
                        square_bottom_right2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    gabor_patch.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in gabor_patch.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "gabor_patch" ---
            for thisComponent in gabor_patch.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for gabor_patch
            gabor_patch.tStop = globalClock.getTime(format='float')
            gabor_patch.tStopRefresh = tThisFlipGlobal
            thisExp.addData('gabor_patch.stopped', gabor_patch.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if gabor_patch.maxDurationReached:
                routineTimer.addTime(-gabor_patch.maxDuration)
            elif gabor_patch.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.100000)
            
            # --- Prepare to start Routine "response" ---
            # create an object to store info about Routine response
            response = data.Routine(
                name='response',
                components=[key_resp],
            )
            response.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # store start times for response
            response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            response.tStart = globalClock.getTime(format='float')
            response.status = STARTED
            thisExp.addData('response.started', response.tStart)
            response.maxDuration = None
            # keep track of which components have finished
            responseComponents = response.components
            for thisComponent in response.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "response" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            response.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp is stopping this frame...
                if key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp.tStop = t  # not accounting for scr refresh
                        key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.stopped')
                        # update status
                        key_resp.status = FINISHED
                        key_resp.status = FINISHED
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                        # was this correct?
                        if (key_resp.keys == str(corr_response)) or (key_resp.keys == corr_response):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    response.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in response.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "response" ---
            for thisComponent in response.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for response
            response.tStop = globalClock.getTime(format='float')
            response.tStopRefresh = tThisFlipGlobal
            thisExp.addData('response.stopped', response.tStop)
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(corr_response).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('key_resp.keys',key_resp.keys)
            trials.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                trials.addData('key_resp.rt', key_resp.rt)
                trials.addData('key_resp.duration', key_resp.duration)
            # Run 'End Routine' code from code_4
            if str(key_resp.keys) == str(corr_response):
                key_resp.corr = 1
            else:
                key_resp.corr = 0
            
            thisExp.addData('gabor_angle', gabor_angle)
            thisExp.addData('level', level)
            thisExp.addData('jitter', jitter)
            thisExp.addData('tiltDir', tiltDir)
            thisExp.addData('key_resp.corr', key_resp.corr)
            
            # Per-block tracking
            if key_resp.keys in ['', [], None]:
                expInfo[f'block_{current_block}_noresp'] += 1
            else:
                expInfo[f'block_{current_block}_total'] += 1
                if key_resp.corr == 1:
                    expInfo[f'block_{current_block}_correct'] += 1
            
            # Cumulative tracking
            if key_resp.keys not in ['', [], None]:
                expInfo['total_trials'] += 1
                if key_resp.corr == 1:
                    expInfo['total_correct'] += 1
            else:
                expInfo['total_noresp'] += 1
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if response.maxDurationReached:
                routineTimer.addTime(-response.maxDuration)
            elif response.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "eye_stop" ---
        # create an object to store info about Routine eye_stop
        eye_stop = data.Routine(
            name='eye_stop',
            components=[etRecord_stop],
        )
        eye_stop.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_eye_stop_trigger
        win.callOnFlip(eyetracker.sendMessage, f"BLOCK_{current_block}_REC_STOP")
        thisExp.addData('eyetracker_rec_stop_block', current_block)
        # store start times for eye_stop
        eye_stop.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        eye_stop.tStart = globalClock.getTime(format='float')
        eye_stop.status = STARTED
        thisExp.addData('eye_stop.started', eye_stop.tStart)
        eye_stop.maxDuration = None
        # keep track of which components have finished
        eye_stopComponents = eye_stop.components
        for thisComponent in eye_stop.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "eye_stop" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        eye_stop.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *etRecord_stop* updates
            if etRecord_stop.status == NOT_STARTED:
                etRecord_stop.frameNStart = frameN  # exact frame index
                etRecord_stop.tStart = t  # local t and not account for scr refresh
                etRecord_stop.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(etRecord_stop, 'tStartRefresh')  # time at next scr refresh
                etRecord_stop.status = STARTED
            
            # if etRecord_stop is stopping this frame...
            if etRecord_stop.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > etRecord_stop.tStartRefresh + 0.0-frameTolerance:
                    # keep track of stop time/frame for later
                    etRecord_stop.tStop = t  # not accounting for scr refresh
                    etRecord_stop.tStopRefresh = tThisFlipGlobal  # on global time
                    etRecord_stop.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('etRecord_stop.stopped', t)
                    # update status
                    etRecord_stop.status = FINISHED
                    etRecord_stop.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                eye_stop.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in eye_stop.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "eye_stop" ---
        for thisComponent in eye_stop.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for eye_stop
        eye_stop.tStop = globalClock.getTime(format='float')
        eye_stop.tStopRefresh = tThisFlipGlobal
        thisExp.addData('eye_stop.stopped', eye_stop.tStop)
        # the Routine "eye_stop" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "between_block" ---
        # create an object to store info about Routine between_block
        between_block = data.Routine(
            name='between_block',
            components=[end_block_text, between_block_resp],
        )
        between_block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        end_block_text.setText(f"Block {current_block} of {target_blocks} complete!\n\nPlease wait for the experimenter to advance.")
        # create starting attributes for between_block_resp
        between_block_resp.keys = []
        between_block_resp.rt = []
        _between_block_resp_allKeys = []
        # Run 'Begin Routine' code from code_10
        # Write the block summary immediately when the block ends, rather than
        # waiting for the experimenter to press SPACE to advance - this guarantees
        # the summary is captured even if the run stops on this screen.
        block_correct = expInfo[f'block_{current_block}_correct']
        block_total = expInfo[f'block_{current_block}_total']
        block_noresp = expInfo[f'block_{current_block}_noresp']
        
        if block_total > 0:
            block_acc = (block_correct / block_total) * 100
        else:
            block_acc = 0
        
        print(f"\nBlock {current_block} Summary:")
        print(f"  Trials completed: {block_total}")
        print(f"  No-response trials: {block_noresp}")
        print(f"  Correct responses: {block_correct}")
        print(f"  Block accuracy: {block_acc:.1f}%")
        
        # Save block summary to data output
        thisExp.addData('block_number', current_block)
        thisExp.addData('block_total_trials', block_total)
        thisExp.addData('block_correct', block_correct)
        thisExp.addData('block_noresp', block_noresp)
        thisExp.addData('block_accuracy', block_acc)
        thisExp.addData('threshold_angle_used', level)
        
        # Cumulative totals as of this block, saved here (not just at the very end)
        # so they survive even if the run stops before reaching the final routine
        overall_accuracy = (expInfo['total_correct'] / expInfo['total_trials'] * 100) if expInfo['total_trials'] > 0 else 0
        thisExp.addData('total_blocks_completed', current_block)
        thisExp.addData('total_trials', expInfo['total_trials'])
        thisExp.addData('total_correct', expInfo['total_correct'])
        thisExp.addData('total_noresp', expInfo['total_noresp'])
        thisExp.addData('overall_accuracy', overall_accuracy)
        
        # Increment block counter
        current_block += 1
        
        # Check if we've hit target blocks and ask experimenter
        if current_block > target_blocks:
            print(f"\nTarget of {target_blocks} blocks reached.")
            print("Experimenter can press SPACE to run another block or ESC to end.")
        # store start times for between_block
        between_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        between_block.tStart = globalClock.getTime(format='float')
        between_block.status = STARTED
        thisExp.addData('between_block.started', between_block.tStart)
        between_block.maxDuration = None
        # keep track of which components have finished
        between_blockComponents = between_block.components
        for thisComponent in between_block.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "between_block" ---
        # if trial has changed, end Routine now
        if isinstance(blocks, data.TrialHandler2) and thisBlock.thisN != blocks.thisTrial.thisN:
            continueRoutine = False
        between_block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *end_block_text* updates
            
            # if end_block_text is starting this frame...
            if end_block_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_block_text.frameNStart = frameN  # exact frame index
                end_block_text.tStart = t  # local t and not account for scr refresh
                end_block_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_block_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_block_text.started')
                # update status
                end_block_text.status = STARTED
                end_block_text.setAutoDraw(True)
            
            # if end_block_text is active this frame...
            if end_block_text.status == STARTED:
                # update params
                pass
            
            # *between_block_resp* updates
            waitOnFlip = False
            
            # if between_block_resp is starting this frame...
            if between_block_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                between_block_resp.frameNStart = frameN  # exact frame index
                between_block_resp.tStart = t  # local t and not account for scr refresh
                between_block_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(between_block_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'between_block_resp.started')
                # update status
                between_block_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(between_block_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(between_block_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if between_block_resp.status == STARTED and not waitOnFlip:
                theseKeys = between_block_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _between_block_resp_allKeys.extend(theseKeys)
                if len(_between_block_resp_allKeys):
                    between_block_resp.keys = _between_block_resp_allKeys[-1].name  # just the last key pressed
                    between_block_resp.rt = _between_block_resp_allKeys[-1].rt
                    between_block_resp.duration = _between_block_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                between_block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in between_block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "between_block" ---
        for thisComponent in between_block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for between_block
        between_block.tStop = globalClock.getTime(format='float')
        between_block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('between_block.stopped', between_block.tStop)
        # check responses
        if between_block_resp.keys in ['', [], None]:  # No response was made
            between_block_resp.keys = None
        blocks.addData('between_block_resp.keys',between_block_resp.keys)
        if between_block_resp.keys != None:  # we had a response
            blocks.addData('between_block_resp.rt', between_block_resp.rt)
            blocks.addData('between_block_resp.duration', between_block_resp.duration)
        # the Routine "between_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 999.0 repeats of 'blocks'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "endroutine" ---
    # create an object to store info about Routine endroutine
    endroutine = data.Routine(
        name='endroutine',
        components=[text],
    )
    endroutine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for endroutine
    endroutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    endroutine.tStart = globalClock.getTime(format='float')
    endroutine.status = STARTED
    thisExp.addData('endroutine.started', endroutine.tStart)
    endroutine.maxDuration = None
    # keep track of which components have finished
    endroutineComponents = endroutine.components
    for thisComponent in endroutine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "endroutine" ---
    endroutine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + jitter-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.tStopRefresh = tThisFlipGlobal  # on global time
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            endroutine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endroutine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "endroutine" ---
    for thisComponent in endroutine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for endroutine
    endroutine.tStop = globalClock.getTime(format='float')
    endroutine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('endroutine.stopped', endroutine.tStop)
    # Run 'End Routine' code from code_7
    # EyeLink cleanup is handled automatically by PsychoPy
    win.mouseVisible = True
    
    # Print final summary across all blocks
    print(f"\n{'='*50}")
    print(f"EXPERIMENT COMPLETE - FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total blocks completed: {current_block - 1}")
    print(f"Total trials: {expInfo['total_trials']}")
    print(f"Total no-response trials: {expInfo['total_noresp']}")
    
    if expInfo['total_trials'] > 0:
        overall_acc = (expInfo['total_correct'] / expInfo['total_trials']) * 100
    else:
        overall_acc = 0
    
    print(f"Overall accuracy: {overall_acc:.1f}%")
    print(f"Threshold angle used: {level:.2f} degrees")
    
    print(f"\nPer-block breakdown:")
    
    for b in range(1, current_block):
        b_correct = expInfo.get(f'block_{b}_correct', 0)
        b_total = expInfo.get(f'block_{b}_total', 0)
        b_noresp = expInfo.get(f'block_{b}_noresp', 0)
        b_acc = (b_correct / b_total * 100) if b_total > 0 else 0
        print(f"  Block {b}: {b_total} trials, "
              f"{b_noresp} no-resp, "
              f"{b_acc:.1f}% accuracy")
    
    print(f"{'='*50}\n") 
    thisExp.nextEntry()
    # the Routine "endroutine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from code_7
    print(f"Experiment complete!")
    print(f"Threshold angle used: {level:.2f} degrees")
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
