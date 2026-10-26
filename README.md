## 6. Scripts, and what each one produces

    milestone_head.py        label-count scaling on the 12-study split
    final_eval.py            the headline 2x2: 0.609 -> 0.733  (PRIMARY and
                             SECONDARY metrics)      -> weak/final_cv_results.json
    final_model.py           ships the sagittal 0.733 checkpoint
    final_model_axial.py     ships the axial checkpoint
    final_model_b2.py        ships the 3-plane head
    lever_a.py               slice aggregators       -> weak/lever_a_results.json
    lever_b.py               planes; env NWEAK / NOFLAG switch the 2x2 and the
                             flag control            -> weak/lever_b_*.json
    lever_c.py               backbone fine-tune, WRITTEN AND SMOKE-TESTED, NOT
                             RUN (see REPORT.md section on cost)
    lever_d.py               Adrita's BiLSTM, her class verbatim
                                                     -> weak/lever_d_3plane.json
    lever_e.py               frozen-backbone sweep   -> weak/lever_e_backbones.json
    extract_*.py             the feature caches above
    inference.py             loads a head, writes submission.csv
