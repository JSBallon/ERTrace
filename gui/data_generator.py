'''
    st.markdown("#### Generate Faker Datasets")
    c_m, c_n, c_ov, c_nr, c_tr = st.columns([1.2, 1.2, 1.6, 1.4, 1.4])
    with c_m:
        n_faker_a = st.number_input("Source A entries (M)", min_value=10,
                                    max_value=500, value=100, step=10)
    with c_n:
        n_faker_b = st.number_input("Source B entries (N)", min_value=10,
                                    max_value=500, value=120, step=10)
    with c_ov:
        overlap_pct_ui = st.slider("Overlap %", min_value=0, max_value=100,
                                   value=80, step=5)
    with c_nr:
        noise_rate_ui = st.slider("Noise rate %", min_value=0, max_value=100,
                                  value=30, step=5)
    with c_tr:
        typo_rate_ui = st.slider("Typo rate %", min_value=0, max_value=100,
                                 value=30, step=5)

    # Derived stats
    k_intended = int(n_faker_a * overlap_pct_ui / 100)
    k_actual   = min(k_intended, n_faker_b)
    if k_actual < k_intended:
        st.warning(
            f"Overlap {overlap_pct_ui}% requires {k_intended} shared entries "
            f"but N={n_faker_b} — capped at {k_actual} ({k_actual*100//n_faker_a}%)."
        )
    st.caption(
        f"→ **{k_actual}** shared · "
        f"**{n_faker_a - k_actual}** A-only · "
        f"**{n_faker_b - k_actual}** B-only"
    )

    faker_button = st.button("Generate Faker Datasets", type="primary",
                             use_container_width=False)
    if faker_button:
        gen = FakerDataGenerator()
        records_a, records_b = gen.generate_paired_datasets(
            n_a=n_faker_a,
            n_b=n_faker_b,
            overlap_pct=overlap_pct_ui / 100,
            noise_rate=noise_rate_ui / 100,
            typo_rate=typo_rate_ui / 100,
        )
        path_a = _write_input_csv(records_a, "source_a")
        path_b = _write_input_csv(records_b, "source_b")
        st.session_state.temp_a_path = path_a
        st.session_state.temp_b_path = path_b
        st.session_state.faker_k     = k_actual
        st.success(
            f"Generated Source A ({n_faker_a} entries) and "
            f"Source B ({n_faker_b} entries) with {k_actual} shared entries."
        )

    st.divider()
    '''