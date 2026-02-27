# QC Issue Types

Explanations of every `issue_type` value that can appear in `qc_report.csv`.

---

## clipping

**What it is**  
Clipping occurs when a signal's amplitude reaches or exceeds the maximum value the recording system can represent (±1.0 in normalised floating-point). Any sample at or above the `CLIPPING_THRESHOLD` (default `0.998`) is considered clipped.

**What it sounds like**  
A harsh, distorted crunch or buzz. Severe clipping sounds like crackle or digital breakup. Mild clipping on a single transient can be almost inaudible but still represents data loss.

**What causes it in audiobooks**  
- Microphone preamp gain set too high
- Narrator speaking too close to the mic during loud passages
- Post-processing (compression, normalisation) that pushes peaks into the ceiling

**Severity score**  
Number of continuously clipped samples in the region. A score of `1` is a single sample touching the ceiling; a score of `500` is a sustained saturated region roughly 11 ms long at 44.1 kHz. Higher = worse.

**What to do**  
Check the flagged timestamp in your DAW. If it is a brief transient peak, a limiter or gentle gain reduction at that point may be enough. Sustained clipping usually requires a re-record.

---

## click

**What it is**  
A click is a very short, high-frequency transient spike — typically lasting less than `CLICK_MAX_DURATION_MS` (default 15 ms). Detected by band-pass filtering to the `CLICK_BAND` (default 2,500–10,000 Hz) and flagging frames where the RMS energy spikes to more than `CLICK_THRESHOLD_MULTIPLIER` (default 5×) above the rolling median baseline.

**What it sounds like**  
A sharp tick, pop, or snap. Can range from a faint electrical tick to a loud mouth click or hard consonant pop.

**What causes it in audiobooks**  
- Mouth sounds and lip smacks
- Saliva clicks between words
- Electrical interference (cable faults, ground loops)
- Mouse clicks or keyboard taps picked up by the mic
- Digital glitches from sample-rate mismatches or buffer dropouts

**Severity score**  
Peak ratio of `frame_rms / rolling_median_rms` at the transient. A score of `5.0` means the click was exactly at the detection threshold; a score of `20.0` means it was four times louder than threshold. Higher = more audible.

**What to do**  
Open the timestamp in your DAW and zoom in. Mouth clicks and lip smacks can often be manually silenced or attenuated with a draw tool. Electrical clicks may need a de-click plugin (e.g. iZotope RX — Spectral Repair or De-click).

---

## plosive

**What it is**  
A plosive is a low-frequency pressure burst caused by consonant sounds (P, B, T, D, K, G) hitting the microphone diaphragm. Detected by band-pass filtering to the `PLOSIVE_BAND` (default 20–180 Hz) and flagging short bursts that exceed `PLOSIVE_THRESHOLD_MULTIPLIER` (default 4×) the rolling median RMS, lasting less than `PLOSIVE_MAX_DURATION_MS` (default 80 ms).

**What it sounds like**  
A low-frequency thump, boom, or "wump" — most obvious on words starting with P or B. Can cause the audio to momentarily distort or the listener to feel a physical pressure sensation.

**What causes it in audiobooks**  
- Narrator speaking directly into the microphone without a pop filter
- Pop filter placed too far from the mic or at an angle
- Narrator turning toward the mic on impact consonants
- High microphone sensitivity at low frequencies

**Severity score**  
Peak ratio of `frame_rms / rolling_median_rms` in the plosive band. Higher = larger low-frequency pressure burst = more likely to be audible and disturbing.

**What to do**  
Light plosives can often be attenuated with a high-pass filter applied only to the flagged region (typical cutoff 80–120 Hz). Severe plosives usually require manual gain reduction or a re-record of the affected sentence. iZotope RX's De-plosive module is purpose-built for this.

---

## noise_floor_jump

**What it is**  
A sudden change in the background noise level between two adjacent measurement windows. Detected by computing RMS in windows of `NOISE_WINDOW_SECONDS` (default 1.0 s), converting to dBFS, and flagging transitions where `|ΔdB| ≥ NOISE_DB_DELTA_THRESHOLD` (default 14 dB), with a minimum separation of `NOISE_MIN_SEPARATION_SECONDS` (default 10 s) between reported events.

**What it sounds like**  
A sudden audible shift in the background hiss or room tone — the "ambience" of the recording noticeably changes from one moment to the next. On headphones this can feel like the room suddenly gets quieter or louder.

**What causes it in audiobooks**  
- Recording sessions stitched together from different takes recorded at different times or in slightly different room conditions
- Air conditioning, HVAC, or traffic noise cutting in or out
- A different microphone or preamp gain setting used for a re-record
- Long pauses where background noise varies (e.g. a phone notification silenced mid-session)
- The narrator moving between takes without maintaining a consistent distance from the mic

**Severity score**  
The absolute dB difference between the two adjacent windows (`|ΔdB|`). A score of `14.0` is exactly at the detection threshold. A score of `28.0` means the noise floor shifted by nearly 30 dB — very likely audible to any listener.

**What to do**  
Find the edit point at the reported timestamp and listen to the join. If it is a stitch between takes, adding a short crossfade or manually matching the room tone between segments will fix it. iZotope RX's Ambience Match or Adobe Audition's Match Loudness tools can automate the levelling. If it is caused by environmental noise, Noise Reduction applied to the louder section may help.

---

## Severity score — general guidance

| Range | Interpretation |
|---|---|
| At threshold | Marginal — worth noting, may be inaudible |
| 2–5× threshold | Moderate — likely audible on headphones |
| 5–10× threshold | Significant — audible on any playback system |
| 10×+ threshold | Severe — will be noticed by any listener |

Tier assignment (`primary` / `secondary`) reflects global rank across all files in the batch, not absolute severity. A `secondary` event in a clean batch may still be worth addressing; a `secondary` event in a very noisy batch may be below the threshold of concern.
