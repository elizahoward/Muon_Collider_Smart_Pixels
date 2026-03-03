# This script segment is generated automatically by AutoPilot

# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 87 \
    name data1_0_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data1_0_val \
    op interface \
    ports { data1_0_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 88 \
    name data1_1_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data1_1_val \
    op interface \
    ports { data1_1_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 89 \
    name data1_2_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data1_2_val \
    op interface \
    ports { data1_2_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 90 \
    name data1_3_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data1_3_val \
    op interface \
    ports { data1_3_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 91 \
    name data1_4_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data1_4_val \
    op interface \
    ports { data1_4_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 92 \
    name data1_5_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data1_5_val \
    op interface \
    ports { data1_5_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 93 \
    name data1_6_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data1_6_val \
    op interface \
    ports { data1_6_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 94 \
    name data1_7_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data1_7_val \
    op interface \
    ports { data1_7_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 95 \
    name data2_0_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data2_0_val \
    op interface \
    ports { data2_0_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 96 \
    name data2_1_val \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_data2_1_val \
    op interface \
    ports { data2_1_val { I 8 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_ready { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -2 \
    name ap_return \
    type ap_return \
    reset_level 1 \
    sync_rst true \
    corename ap_return \
    op interface \
    ports { ap_return { O 1 vector } } \
} "
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


