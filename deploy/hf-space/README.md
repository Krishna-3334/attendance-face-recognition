---
title: BioAccess Face Attendance API
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# BioAccess — face-recognition attendance API

Backend for [attendance-face-recognition](https://github.com/Krishna-3334/attendance-face-recognition).
Detection with SCRFD-10GF, 512-D ArcFace embeddings, and a MiniFASNet passive
liveness check that runs before any gallery comparison, so a printed photo or a
phone screen cannot produce an attendance record.

Interactive API docs: `/docs`

The gallery lives on the Space's ephemeral disk, so enrolments are cleared when
the Space restarts — this is a public demo, not a store of anyone's biometrics.
