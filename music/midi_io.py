
# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MIDI ops.

Input and output wrappers for converting between MIDI and other formats.
"""

# Import necessary modules
import collections
import tempfile
from magenta.music import constants
from magenta.music.protobuf import music_pb2
import pretty_midi
import tensorflow.compat.v1 as tf

# Allow pretty_midi to read MIDI files with very high tick rates.
pretty_midi.pretty_midi.MAX_TICK = 1e10

# Offset for converting major to minor keys in PrettyMIDI
_PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET = 12

# Custom exception for MIDI conversion errors
class MIDIConversionError(Exception):
  pass

# Function to convert a sequence of MIDI events to a NoteSequence
def midi_to_note_sequence(meas, midi=False):    
  sequence = music_pb2.NoteSequence()
  start_time = 0
  # Iterate over each pitch-duration pair in the input
  for (pitch, duration) in meas:
    if pitch > 0:
      # Create a new note in the sequence for each pitch
      note = sequence.notes.add()
      note.instrument = 1
      note.program = 1
      note.start_time = start_time
      note.end_time = start_time + duration
      start_time += duration

      note.pitch = pitch
      note.velocity = 127
      note.is_drum = False
    else:
      # If pitch is 0 or less, it's a rest; increment start time
      start_time += duration

  # TODO: Estimate note type and populate note.numerator and note.denominator

  return sequence

# Function to convert a MIDI file to a NoteSequence
def midi_file_to_note_sequence(midi_file):
  """Converts MIDI file to a NoteSequence.

  Args:
    midi_file: A string path to a MIDI file.

  Returns:
    A NoteSequence.

  Raises:
    MIDIConversionError: Invalid midi_file.
  """
  return midi_to_note_sequence(midi_file)

# Function to write a NoteSequence to a MIDI file
def note_sequence_to_midi_file(sequence, output_file,
                               drop_events_n_seconds_after_last_note=None):
  pretty_midi_object = note_sequence_to_pretty_midi(
      sequence, drop_events_n_seconds_after_last_note)
  with tempfile.NamedTemporaryFile() as temp_file:
    pretty_midi_object.write(temp_file)
    temp_file.flush()
    temp_file.seek(0)
    tf.gfile.Copy(temp_file.name, output_file, overwrite=True)

# Function to convert a NoteSequence to a PrettyMIDI object
def note_sequence_to_pretty_midi(
    sequence, drop_events_n_seconds_after_last_note=None):
  ticks_per_quarter = sequence.ticks_per_quarter or constants.STANDARD_PPQ

  max_event_time = None
  if drop_events_n_seconds_after_last_note is not None:
    max_event_time = (max([n.end_time for n in sequence.notes] or [0]) +
                      drop_events_n_seconds_after_last_note)

  # Find initial tempo
  initial_seq_tempo = None
  for seq_tempo in sequence.tempos:
    if seq_tempo.time == 0:
      initial_seq_tempo = seq_tempo
      break

  # Set initial tempo if available
  kwargs = {}
  if initial_seq_tempo:
    kwargs['initial_tempo'] = initial_seq_tempo.qpm
  else:
    kwargs['initial_tempo'] = constants.DEFAULT_QUARTERS_PER_MINUTE

  # Create PrettyMIDI object
  pm = pretty_midi.PrettyMIDI(resolution=ticks_per_quarter, **kwargs)

  # Create an instrument to contain time and key signatures
  instrument = pretty_midi.Instrument(0)
  pm.instruments.append(instrument)

  # Populate time signatures
  for seq_ts in sequence.time_signatures:
    if max_event_time and seq_ts.time > max_event_time:
      continue
    time_signature = pretty_midi.containers.TimeSignature(
        seq_ts.numerator, seq_ts.denominator, seq_ts.time)
    pm.time_signature_changes.append(time_signature)

  # Populate key signatures
  for seq_key in sequence.key_signatures:
    if max_event_time and seq_key.time > max_event_time:
      continue
    key_number = seq_key.key
    if seq_key.mode == seq_key.MINOR:
      key_number += _PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET
    key_signature = pretty_midi.containers.KeySignature(
        key_number, seq_key.time)
    pm.key_signature_changes.append(key_signature)

  # Populate tempos
  for seq_tempo in sequence.tempos:
    if seq_tempo == initial_seq_tempo or (max_event_time and seq_tempo.time > max_event_time):
      continue
    tick_scale = 60.0 / (pm.resolution * seq_tempo.qpm)
    tick = pm.time_to_tick(seq_tempo.time)
    pm._tick_scales.append((tick, tick_scale))
    pm._update_tick_to_time(0)

  # Create a mapping between instrument index and name
  inst_infos = {}
  for inst_info in sequence.instrument_infos:
    inst_infos[inst_info.instrument] = inst_info.name

  # Group instrument events by type
  instrument_events = collections.defaultdict(
      lambda: collections.defaultdict(list))
  for seq_note in sequence.notes:
    instrument_events[(seq_note.instrument, seq_note.program,
                       seq_note.is_drum)]['notes'].append(
                           pretty_midi.Note(
                               seq_note.velocity, seq_note.pitch,
                               seq_note.start_time, seq_note.end_time))
  for seq_bend in sequence.pitch_bends:
    if max_event_time and seq_bend.time > max_event_time:
      continue
    instrument_events[(seq_bend.instrument, seq_bend.program,
                       seq_bend.is_drum)]['bends'].append(
                           pretty_midi.PitchBend(seq_bend.bend, seq_bend.time))
  for seq_cc in sequence.control_changes:
    if max_event_time and seq_cc.time > max_event_time:
      continue
    instrument_events[(seq_cc.instrument, seq_cc.program,
                       seq_cc.is_drum)]['controls'].append(
                           pretty_midi.ControlChange(
                               seq_cc.control_number,
                               seq_cc.control_value, seq_cc.time))

  # Write events to the PrettyMIDI object
  for (instr_id, prog_id, is_drum) in sorted(instrument_events.keys()):
    if instr_id > 0:
      instrument = pretty_midi.Instrument(prog_id, is_drum)
      pm.instruments.append(instrument)
    else:
      instrument.is_drum = is_drum
    instrument.program = prog_id
    if instr_id in inst_infos:
      instrument.name = inst_infos[instr_id]
    instrument.notes = instrument_events[
        (instr_id, prog_id, is_drum)]['notes']
    instrument.pitch_bends = instrument_events[
        (instr_id, prog_id, is_drum)]['bends']
    instrument.control_changes = instrument_events[
        (instr_id, prog_id, is_drum)]['controls']

  return pm

# Renamed function for converting MIDI data to NoteSequence
def midi_to_sequence_proto(midi_data):
  return midi_to_note_sequence(midi_data)

# Renamed function for converting NoteSequence to PrettyMIDI
def sequence_proto_to_pretty_midi(sequence,
                                  drop_events_n_seconds_after_last_note=None):
  return note_sequence_to_pretty_midi(sequence,
                                      drop_events_n_seconds_after_last_note)

# Renamed function for converting MIDI file to NoteSequence
def midi_file_to_sequence_proto(midi_file):
  return midi_file_to_note_sequence(midi_file)

# Renamed function for writing NoteSequence to a MIDI file
def sequence_proto_to_midi_file(sequence, output_file,
                                drop_events_n_seconds_after_last_note=None):
  return note_sequence_to_midi_file(sequence, output_file,
                                    drop_events_n_seconds_after_last_note)
