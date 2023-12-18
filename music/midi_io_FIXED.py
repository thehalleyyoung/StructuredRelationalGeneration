
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

# Import necessary Python libraries for handling MIDI files and related operations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
import tempfile
import io
from magenta.music import constants
from magenta.music.protobuf import music_pb2
import pretty_midi
import six
import tensorflow.compat.v1 as tf

# Increase the maximum number of ticks for pretty_midi to handle large datasets
pretty_midi.pretty_midi.MAX_TICK = 1e10

# Offset to convert between major and minor keys in PrettyMIDI
_PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET = 12


class MIDIConversionError(Exception):
  """Custom exception for errors during MIDI conversion."""
  pass

def midi_to_note_sequence(midi_data):
  """Converts MIDI data to a NoteSequence object.

  Args:
    midi_data: A byte string containing the contents of a MIDI file or an
        already populated pretty_midi.PrettyMIDI object.

  Returns:
    A music_pb2.NoteSequence object representing the MIDI data.

  Raises:
    MIDIConversionError: If an error occurs during conversion.
  """
  # Load MIDI data into PrettyMIDI object
  midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_data))
  # Create an empty NoteSequence
  sequence = music_pb2.NoteSequence()

  # Populate the NoteSequence header with MIDI metadata
  sequence.ticks_per_quarter = midi.resolution
  sequence.source_info.parser = music_pb2.NoteSequence.SourceInfo.PRETTY_MIDI
  sequence.source_info.encoding_type = (
      music_pb2.NoteSequence.SourceInfo.MIDI)

  # Convert MIDI time signatures to NoteSequence time signatures
  for midi_time in midi.time_signature_changes:
    time_signature = sequence.time_signatures.add()
    time_signature.time = midi_time.time
    time_signature.numerator = midi_time.numerator
    try:
      time_signature.denominator = midi_time.denominator
    except ValueError:
      raise MIDIConversionError('Invalid time signature denominator %d' %
                                midi_time.denominator)

  # Convert MIDI key signatures to NoteSequence key signatures
  for midi_key in midi.key_signature_changes:
    key_signature = sequence.key_signatures.add()
    key_signature.time = midi_key.time
    key_signature.key = midi_key.key_number % 12
    midi_mode = midi_key.key_number // 12
    if midi_mode == 0:
      key_signature.mode = key_signature.MAJOR
    elif midi_mode == 1:
      key_signature.mode = key_signature.MINOR
    else:
      raise MIDIConversionError('Invalid midi_mode %i' % midi_mode)

  # Convert MIDI tempo changes to NoteSequence tempo changes
  tempo_times, tempo_qpms = midi.get_tempo_changes()
  for time_in_seconds, tempo_in_qpm in zip(tempo_times, tempo_qpms):
    tempo = sequence.tempos.add()
    tempo.time = time_in_seconds
    tempo.qpm = tempo_in_qpm

  # Gather all notes, pitch bends, and control changes from all instruments
  midi_notes = []
  midi_pitch_bends = []
  midi_control_changes = []
  for num_instrument, midi_instrument in enumerate(midi.instruments):
    if midi_instrument.name:
      instrument_info = sequence.instrument_infos.add()
      instrument_info.name = midi_instrument.name
      instrument_info.instrument = num_instrument
    for midi_note in midi_instrument.notes:
      if not sequence.total_time or midi_note.end > sequence.total_time:
        sequence.total_time = midi_note.end
      midi_notes.append((midi_instrument.program, num_instrument,
                         midi_instrument.is_drum, midi_note))
    for midi_pitch_bend in midi_instrument.pitch_bends:
      midi_pitch_bends.append(
          (midi_instrument.program, num_instrument,
           midi_instrument.is_drum, midi_pitch_bend))
    for midi_control_change in midi_instrument.control_changes:
      midi_control_changes.append(
          (midi_instrument.program, num_instrument,
           midi_instrument.is_drum, midi_control_change))

  # Add notes to the NoteSequence
  for program, instrument, is_drum, midi_note in midi_notes:
    note = sequence.notes.add()
    note.instrument = instrument
    note.program = program
    note.start_time = midi_note.start
    note.end_time = midi_note.end
    note.pitch = midi_note.pitch
    note.velocity = midi_note.velocity
    note.is_drum = is_drum

  # Add pitch bends to the NoteSequence
  for program, instrument, is_drum, midi_pitch_bend in midi_pitch_bends:
    pitch_bend = sequence.pitch_bends.add()
    pitch_bend.instrument = instrument
    pitch_bend.program = program
    pitch_bend.time = midi_pitch_bend.time
    pitch_bend.bend = midi_pitch_bend.pitch
    pitch_bend.is_drum = is_drum

  # Add control changes to the NoteSequence
  for program, instrument, is_drum, midi_control_change in midi_control_changes:
    control_change = sequence.control_changes.add()
    control_change.instrument = instrument
    control_change.program = program
    control_change.time = midi_control_change.time
    control_change.control_number = midi_control_change.number
    control_change.control_value = midi_control_change.value
    control_change.is_drum = is_drum

  # Future work: Estimate the note type (e.g., quarter note) and populate note.numerator and note.denominator

  return sequence


def midi_file_to_note_sequence(midi_file):
  """Converts a MIDI file to a NoteSequence.

  Args:
    midi_file: A string path to a MIDI file.

  Returns:
    A NoteSequence.

  Raises:
    MIDIConversionError: If the MIDI file is invalid or cannot be read.
  """
  with open(midi_file, 'rb') as f:
    midi_as_string = f.read()
    return midi_to_note_sequence(midi_as_string)


def note_sequence_to_midi_file(sequence, output_file,
                               drop_events_n_seconds_after_last_note=None):
  """Converts a NoteSequence to a MIDI file and writes it to disk.

  Args:
    sequence: A NoteSequence to be converted.
    output_file: String path to the output MIDI file.
    drop_events_n_seconds_after_last_note: Optional; drop events occurring this
        number of seconds after the last note.

  Note: Time is stored in absolute values in the NoteSequence and is retained
  when converted back to MIDI. The tempo map is recreated in the MIDI file.
  """
  pretty_midi_object = note_sequence_to_pretty_midi(
      sequence, drop_events_n_seconds_after_last_note)
  with tempfile.NamedTemporaryFile() as temp_file:
    pretty_midi_object.write(temp_file)
    temp_file.flush()
    temp_file.seek(0)
    tf.gfile.Copy(temp_file.name, output_file, overwrite=True)


def note_sequence_to_pretty_midi(
    sequence, drop_events_n_seconds_after_last_note=None):
  """Converts a NoteSequence to a PrettyMIDI object.

  Args:
    sequence: A NoteSequence to be converted.
    drop_events_n_seconds_after_last_note: Optional; drop events occurring this
        number of seconds after the last note.

  Returns:
    A pretty_midi.PrettyMIDI object representing the NoteSequence.

  Note: Time is stored in absolute values in the NoteSequence and is retained
  when converted back to PrettyMIDI. The tempo map is recreated in the PrettyMIDI object.
  """
  # Set the ticks per quarter note (resolution) for the PrettyMIDI object
  ticks_per_quarter = sequence.ticks_per_quarter or constants.STANDARD_PPQ

  # Calculate the maximum time for events if specified
  max_event_time = None
  if drop_events_n_seconds_after_last_note is not None:
    max_event_time = (max([n.end_time for n in sequence.notes] or [0]) +
                      drop_events_n_seconds_after_last_note)

  # Try to find the initial tempo of the sequence
  initial_seq_tempo = None
  for seq_tempo in sequence.tempos:
    if seq_tempo.time == 0:
      initial_seq_tempo = seq_tempo
      break

  # Set the initial tempo for the PrettyMIDI object
  kwargs = {'initial_tempo': initial_seq_tempo.qpm} if initial_seq_tempo else {'initial_tempo': constants.DEFAULT_QUARTERS_PER_MINUTE}

  # Create a PrettyMIDI object with the specified resolution and initial tempo
  pm = pretty_midi.PrettyMIDI(resolution=ticks_per_quarter, **kwargs)

  # Create an instrument with program 0 to hold time and key signatures
  instrument = pretty_midi.Instrument(0)
  pm.instruments.append(instrument)

  # Add time signatures to the PrettyMIDI object
  for seq_ts in sequence.time_signatures:
    if max_event_time and seq_ts.time > max_event_time:
      continue
    time_signature = pretty_midi.containers.TimeSignature(
        seq_ts.numerator, seq_ts.denominator, seq_ts.time)
    pm.time_signature_changes.append(time_signature)

  # Add key signatures to the PrettyMIDI object
  for seq_key in sequence.key_signatures:
    if max_event_time and seq_key.time > max_event_time:
      continue
    key_number = seq_key.key
    if seq_key.mode == seq_key.MINOR:
      key_number += _PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET
    key_signature = pretty_midi.containers.KeySignature(
        key_number, seq_key.time)
    pm.key_signature_changes.append(key_signature)

  # Add tempo changes to the PrettyMIDI object
  for seq_tempo in sequence.tempos:
    if seq_tempo == initial_seq_tempo:
      continue
    if max_event_time and seq_tempo.time > max_event_time:
      continue
    tick_scale = 60.0 / (pm.resolution * seq_tempo.qpm)
    tick = pm.time_to_tick(seq_tempo.time)
    pm._tick_scales.append((tick, tick_scale))
    pm._update_tick_to_time(0)

  # Map instrument names from the NoteSequence to PrettyMIDI
  inst_infos = {inst_info.instrument: inst_info.name for inst_info in sequence.instrument_infos}

  # Gather and sort instrument events from the NoteSequence
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

  # Add the gathered events to the PrettyMIDI instruments
  for (instr_id, prog_id, is_drum) in sorted(instrument_events.keys()):
    if instr_id > 0:
      instrument = pretty_midi.Instrument(prog_id, is_drum)
      pm.instruments.append(instrument)
    else:
      instrument.is_drum = is_drum
    if instr_id in inst_infos:
      instrument.name = inst_infos[instr_id]
    instrument.notes = instrument_events[
        (instr_id, prog_id, is_drum)]['notes']
    instrument.pitch_bends = instrument_events[
        (instr_id, prog_id, is_drum)]['bends']
    instrument.control_changes = instrument_events[
        (instr_id, prog_id, is_drum)]['controls']

  return pm


def midi_to_sequence_proto(midi_data):
  """Converts MIDI data to a NoteSequence (deprecated name)."""
  return midi_to_note_sequence(midi_data)


def sequence_proto_to_pretty_midi(sequence,
                                  drop_events_n_seconds_after_last_note=None):
  """Converts a NoteSequence to a PrettyMIDI object (deprecated name)."""
  return note_sequence_to_pretty_midi(sequence,
                                      drop_events_n_seconds_after_last_note)


def midi_file_to_sequence_proto(midi_file):
  """Converts a MIDI file to a NoteSequence (deprecated name)."""
  return midi_file_to_note_sequence(midi_file)


def sequence_proto_to_midi_file(sequence, output_file,
                                drop_events_n_seconds_after_last_note=None):
  """Converts a NoteSequence to a MIDI file (deprecated name)."""
  return note_sequence_to_midi_file(sequence, output_file,
                                    drop_events_n_seconds_after_last_note)
