# Labeling rubric (fixed 2026-10-21, used for all Claude report labels)
p in {0, 0.25, 0.5, 0.75, 1} per target. Based on FINDINGS+IMPRESSION.
- ACL: tear/rupture/avulsion=1; grade1-2 sprain or "high-grade partial"=0.75;
  mucoid/cystic degeneration or "possible small partial"=0.5; intact=0.
- MCL: tear incl. partial/grade>=2=1..0.75; grade1 sprain=0.5-0.75;
  periligamentous fluid only=0.25; intact/unmentioned=0. (MPFL/retinaculum
  and LCL do NOT count.)
- Menisci (each side): tear/maceration/root avulsion=1; "suspected/possible
  tear", meniscocapsular separation=0.5-0.75; degeneration/intrasubstance
  signal or extrusion without tear=0.25-0.5; normal/unmentioned=0.
- OA per compartment (Medial/Lateral femorotibial, PF): full-thickness or
  high-grade chondral loss/ulcer, "OA", joint-space narrowing=1; moderate
  thinning/grade2-3=0.75; superficial/low-grade/grade1=0.5; intact=0.
  Trochlea+patella -> PF. "Tricompartmental OA/incipient"=0.75 each.
- Effusion: large/moderate=1; small/mild ("leve")=0.75; trace/minimal=0.5;
  "no significant"/none=0. Conclusion wins over findings on conflict.
- Synovitis: synovial thickening/hypertrophy/synovitis stated=0.75-1;
  bursitis alone does NOT count; unmentioned=0.
- Baker's: popliteal/Baker cyst any size=0.75-1; none/unmentioned=0.
- Contusion: bone marrow edema/bruise (traumatic or peri-fracture)=1;
  degenerative subchondral BME under chondral lesions=0.25-0.5;
  explicitly none=0. Muscle contusion does not count.
- Fracture: any fracture incl. subchondral/insufficiency/impaction/
  osteochondral/avulsion=1 (suspected/"possible"=0.5-0.75); none=0.
- Unmentioned structure in an otherwise thorough report = 0; in a very
  short report = 0 unless pattern suggests otherwise (then 0.25).
