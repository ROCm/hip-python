rocm.bindings.amdsmi
====================

.. py:module:: rocm.bindings.amdsmi


Attributes
----------

.. autoapisummary::

   rocm.bindings.amdsmi.AMDSMI_MAX_MM_IP_COUNT
   rocm.bindings.amdsmi.AMDSMI_MAX_STRING_LENGTH
   rocm.bindings.amdsmi.AMDSMI_MAX_DEVICES
   rocm.bindings.amdsmi.AMDSMI_MAX_CACHE_TYPES
   rocm.bindings.amdsmi.AMDSMI_MAX_ACCELERATOR_PROFILE
   rocm.bindings.amdsmi.AMDSMI_MAX_CP_PROFILE_RESOURCES
   rocm.bindings.amdsmi.AMDSMI_MAX_ACCELERATOR_PARTITIONS
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_NUMA_NODES
   rocm.bindings.amdsmi.AMDSMI_GPU_UUID_SIZE
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK
   rocm.bindings.amdsmi.AMDSMI_MAX_CONTAINER_TYPE
   rocm.bindings.amdsmi.AMDSMI_NUM_HBM_INSTANCES
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_VCN
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_CLKS
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_XGMI_LINKS
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_GFX_CLKS
   rocm.bindings.amdsmi.AMDSMI_MAX_AID
   rocm.bindings.amdsmi.AMDSMI_MAX_ENGINES
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_JPEG
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_JPEG_ENG_V1
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_XCC
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_XCP
   rocm.bindings.amdsmi.AMDSMI_APU_MAX_CORES
   rocm.bindings.amdsmi.AMDSMI_APU_V24_CORES
   rocm.bindings.amdsmi.AMDSMI_APU_MAX_L3
   rocm.bindings.amdsmi.AMDSMI_APU_MAX_IPU
   rocm.bindings.amdsmi.AMDSMI_MAX_NUMBER_OF_AFIDS_PER_RECORD
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_HBM_STACKS
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_AID
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_MID
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_CLKS_PER_AID
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_CLKS_PER_MID
   rocm.bindings.amdsmi.AMDSMI_TIME_FORMAT
   rocm.bindings.amdsmi.AMDSMI_DATE_FORMAT
   rocm.bindings.amdsmi.AMDSMI_LIB_VERSION_MAJOR
   rocm.bindings.amdsmi.AMDSMI_LIB_VERSION_MINOR
   rocm.bindings.amdsmi.AMDSMI_LIB_VERSION_RELEASE
   rocm.bindings.amdsmi.AMDSMI_LIB_VERSION_STRING
   rocm.bindings.amdsmi.AMDSMI_MAX_DRIVER_INFO_RSVD
   rocm.bindings.amdsmi.AMDSMI_MAX_UUID_ELEMENTS
   rocm.bindings.amdsmi.AMDSMI_MAX_SPD_DIMM_ADDRESS
   rocm.bindings.amdsmi.AMDSMI_MAX_SPD_LID
   rocm.bindings.amdsmi.AMDSMI_MAX_SPD_REG_OFFSET
   rocm.bindings.amdsmi.AMDSMI_MAX_SPD_REG_SPACE
   rocm.bindings.amdsmi.AMDSMI_MAX_SPD_WRITE_DATA
   rocm.bindings.amdsmi.AMDSMI_MAX_SVI3_RAIL_INDEX
   rocm.bindings.amdsmi.AMDSMI_MAX_SVI3_RAIL_SELECTION
   rocm.bindings.amdsmi.AMDSMI_POWER_EFFICIENCY_MODE_4
   rocm.bindings.amdsmi.AMDSMI_POWER_EFFICIENCY_MODE_5
   rocm.bindings.amdsmi.AMDSMI_MAX_POWER_EFFICIENCY_UTIL
   rocm.bindings.amdsmi.AMDSMI_MAX_POWER_EFFICIENCY_PPTLIMIT
   rocm.bindings.amdsmi.AMDSMI_RAIL_INDEX_NONE
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_FREQUENCIES
   rocm.bindings.amdsmi.AMDSMI_MAX_FAN_SPEED
   rocm.bindings.amdsmi.AMDSMI_NUM_VOLTAGE_CURVE_POINTS
   rocm.bindings.amdsmi.AMDSMI_MAX_UTILIZATION_VALUES
   rocm.bindings.amdsmi.AMDSMI_MAX_NUM_PM_POLICIES
   rocm.bindings.amdsmi.AMDSMI_MAX_NIC_PORTS
   rocm.bindings.amdsmi.AMDSMI_MAX_NIC_RDMA_DEV
   rocm.bindings.amdsmi.AMDSMI_MAX_NIC_FW
   rocm.bindings.amdsmi.AMDSMI_FABRIC_LABEL_MAX_LENGTH
   rocm.bindings.amdsmi.AMDSMI_FABRIC_PPOD_ID_SIZE
   rocm.bindings.amdsmi.AMDSMI_MAX_CARVEOUT_OPTIONS
   rocm.bindings.amdsmi.processor_type_t


Classes
-------

.. autoapisummary::

   rocm.bindings.amdsmi.timespec
   rocm.bindings.amdsmi.amdsmi_init_flags_t
   rocm.bindings.amdsmi.amdsmi_mm_ip_t
   rocm.bindings.amdsmi.amdsmi_container_types_t
   rocm.bindings.amdsmi.amdsmi_hsmp_driver_version_t
   rocm.bindings.amdsmi.amdsmi_processor_type_t
   rocm.bindings.amdsmi.amdsmi_status_t
   rocm.bindings.amdsmi.amdsmi_clk_type_t
   rocm.bindings.amdsmi.amdsmi_accelerator_partition_type_t
   rocm.bindings.amdsmi.amdsmi_accelerator_partition_resource_type_t
   rocm.bindings.amdsmi.amdsmi_compute_partition_type_t
   rocm.bindings.amdsmi.amdsmi_compute_partition_mem_alloc_mode_t
   rocm.bindings.amdsmi.amdsmi_accelerator_partition_mem_alloc_mode_t
   rocm.bindings.amdsmi.amdsmi_memory_partition_type_t
   rocm.bindings.amdsmi.amdsmi_temperature_type_t
   rocm.bindings.amdsmi.amdsmi_fw_block_t
   rocm.bindings.amdsmi.amdsmi_vram_type_t
   rocm.bindings.amdsmi.amdsmi_range_t
   rocm.bindings.amdsmi.amdsmi_xgmi_info_t
   rocm.bindings.amdsmi.amdsmi_vram_usage_t
   rocm.bindings.amdsmi.amdsmi_violation_status_t
   rocm.bindings.amdsmi.amdsmi_frequency_range_t
   rocm.bindings.amdsmi.amdsmi_bdf_t_bdf_
   rocm.bindings.amdsmi.amdsmi_bdf_t_struct_0
   rocm.bindings.amdsmi.amdsmi_bdf_t
   rocm.bindings.amdsmi.amdsmi_enumeration_info_t
   rocm.bindings.amdsmi.amdsmi_card_form_factor_t
   rocm.bindings.amdsmi.amdsmi_pcie_info_t_pcie_static_
   rocm.bindings.amdsmi.amdsmi_pcie_info_t_pcie_metric_
   rocm.bindings.amdsmi.amdsmi_pcie_info_t
   rocm.bindings.amdsmi.amdsmi_power_cap_info_t
   rocm.bindings.amdsmi.amdsmi_power_cap_type_t
   rocm.bindings.amdsmi.amdsmi_vbios_info_t
   rocm.bindings.amdsmi.amdsmi_cache_property_type_t
   rocm.bindings.amdsmi.amdsmi_gpu_cache_info_t_cache_
   rocm.bindings.amdsmi.amdsmi_gpu_cache_info_t
   rocm.bindings.amdsmi.amdsmi_fw_info_t_fw_info_list_
   rocm.bindings.amdsmi.amdsmi_fw_info_t
   rocm.bindings.amdsmi.amdsmi_asic_info_t
   rocm.bindings.amdsmi.amdsmi_kfd_info_t
   rocm.bindings.amdsmi.amdsmi_nps_caps_t_nps_flags_
   rocm.bindings.amdsmi.amdsmi_nps_caps_t
   rocm.bindings.amdsmi.amdsmi_memory_partition_config_t_numa_range_
   rocm.bindings.amdsmi.amdsmi_memory_partition_config_t
   rocm.bindings.amdsmi.amdsmi_accelerator_partition_profile_t
   rocm.bindings.amdsmi.amdsmi_accelerator_partition_resource_profile_t
   rocm.bindings.amdsmi.amdsmi_accelerator_partition_profile_config_t
   rocm.bindings.amdsmi.amdsmi_link_type_t
   rocm.bindings.amdsmi.amdsmi_cpu_util_t
   rocm.bindings.amdsmi.amdsmi_link_status_t
   rocm.bindings.amdsmi.amdsmi_link_metrics_t__links
   rocm.bindings.amdsmi.amdsmi_link_metrics_t
   rocm.bindings.amdsmi.amdsmi_vram_info_t
   rocm.bindings.amdsmi.amdsmi_driver_info_t
   rocm.bindings.amdsmi.amdsmi_board_info_t
   rocm.bindings.amdsmi.amdsmi_power_info_t
   rocm.bindings.amdsmi.amdsmi_clk_info_t
   rocm.bindings.amdsmi.amdsmi_engine_usage_t
   rocm.bindings.amdsmi.amdsmi_proc_info_t_engine_usage_
   rocm.bindings.amdsmi.amdsmi_proc_info_t_memory_usage_
   rocm.bindings.amdsmi.amdsmi_proc_info_t
   rocm.bindings.amdsmi.amdsmi_proc_gpu_entry_t_struct_0
   rocm.bindings.amdsmi.amdsmi_proc_gpu_entry_t_struct_1
   rocm.bindings.amdsmi.amdsmi_proc_gpu_entry_t
   rocm.bindings.amdsmi.amdsmi_proc_info_by_pid_t
   rocm.bindings.amdsmi.amdsmi_p2p_capability_t
   rocm.bindings.amdsmi.amdsmi_dev_perf_level_t
   rocm.bindings.amdsmi.amdsmi_event_group_t
   rocm.bindings.amdsmi.amdsmi_event_type_t
   rocm.bindings.amdsmi.amdsmi_counter_command_t
   rocm.bindings.amdsmi.amdsmi_counter_value_t
   rocm.bindings.amdsmi.amdsmi_evt_notification_type_t
   rocm.bindings.amdsmi.amdsmi_evt_notification_data_t
   rocm.bindings.amdsmi.amdsmi_temperature_metric_t
   rocm.bindings.amdsmi.amdsmi_voltage_metric_t
   rocm.bindings.amdsmi.amdsmi_voltage_type_t
   rocm.bindings.amdsmi.amdsmi_power_profile_preset_masks_t
   rocm.bindings.amdsmi.amdsmi_gpu_block_t
   rocm.bindings.amdsmi.amdsmi_clk_limit_type_t
   rocm.bindings.amdsmi.amdsmi_cper_sev_t
   rocm.bindings.amdsmi.amdsmi_cper_notify_type_t
   rocm.bindings.amdsmi.amdsmi_gpu_ras_policy_v4_0_t
   rocm.bindings.amdsmi.amdsmi_gpu_ras_policy_info_t_policy_data_
   rocm.bindings.amdsmi.amdsmi_gpu_ras_policy_info_t
   rocm.bindings.amdsmi.amdsmi_ras_err_state_t
   rocm.bindings.amdsmi.amdsmi_memory_type_t
   rocm.bindings.amdsmi.amdsmi_freq_ind_t
   rocm.bindings.amdsmi.amdsmi_xgmi_status_t
   rocm.bindings.amdsmi.amdsmi_memory_page_status_t
   rocm.bindings.amdsmi.amdsmi_utilization_counter_type_t
   rocm.bindings.amdsmi.amdsmi_utilization_counter_t
   rocm.bindings.amdsmi.amdsmi_retired_page_record_t
   rocm.bindings.amdsmi.amdsmi_power_profile_status_t
   rocm.bindings.amdsmi.amdsmi_frequencies_t
   rocm.bindings.amdsmi.amdsmi_dpm_policy_entry_t
   rocm.bindings.amdsmi.amdsmi_dpm_policy_t
   rocm.bindings.amdsmi.amdsmi_pcie_bandwidth_t
   rocm.bindings.amdsmi.amdsmi_version_t
   rocm.bindings.amdsmi.amdsmi_od_vddc_point_t
   rocm.bindings.amdsmi.amdsmi_freq_volt_region_t
   rocm.bindings.amdsmi.amdsmi_od_volt_curve_t
   rocm.bindings.amdsmi.amdsmi_od_volt_freq_data_t
   rocm.bindings.amdsmi.amd_metrics_table_header_t
   rocm.bindings.amdsmi.amdsmi_gpu_xcp_metrics_t
   rocm.bindings.amdsmi.amdsmi_apu_metrics_t
   rocm.bindings.amdsmi.amdsmi_gpu_metrics_t
   rocm.bindings.amdsmi.amdsmi_xgmi_link_status_type_t
   rocm.bindings.amdsmi.amdsmi_xgmi_link_status_t
   rocm.bindings.amdsmi.amdsmi_name_value_t
   rocm.bindings.amdsmi.amdsmi_reg_type_t
   rocm.bindings.amdsmi.amdsmi_ras_feature_t_ras_info_
   rocm.bindings.amdsmi.amdsmi_ras_feature_t
   rocm.bindings.amdsmi.amdsmi_error_count_t
   rocm.bindings.amdsmi.amdsmi_process_info_t
   rocm.bindings.amdsmi.amdsmi_topology_nearest_t
   rocm.bindings.amdsmi.amdsmi_virtualization_mode_t
   rocm.bindings.amdsmi.amdsmi_affinity_scope_t
   rocm.bindings.amdsmi.amdsmi_npm_status_t
   rocm.bindings.amdsmi.amdsmi_npm_info_t
   rocm.bindings.amdsmi.amdsmi_ptl_data_format_t
   rocm.bindings.amdsmi.amdsmi_smu_fw_version_t
   rocm.bindings.amdsmi.amdsmi_ddr_bw_metrics_t
   rocm.bindings.amdsmi.amdsmi_temp_range_refresh_rate_t
   rocm.bindings.amdsmi.amdsmi_dimm_power_t
   rocm.bindings.amdsmi.amdsmi_dimm_thermal_t
   rocm.bindings.amdsmi.amdsmi_io_bw_encoding_t
   rocm.bindings.amdsmi.amdsmi_link_id_bw_type_t
   rocm.bindings.amdsmi.amdsmi_dpm_level_t
   rocm.bindings.amdsmi.amdsmi_hsmp_metrics_table_t
   rocm.bindings.amdsmi.amdsmi_cpu_info_t
   rocm.bindings.amdsmi.amdsmi_sock_info_t
   rocm.bindings.amdsmi.amdsmi_nic_stat_t
   rocm.bindings.amdsmi.amdsmi_nic_asic_info_t
   rocm.bindings.amdsmi.amdsmi_nic_bus_info_t
   rocm.bindings.amdsmi.amdsmi_nic_numa_info_t
   rocm.bindings.amdsmi.amdsmi_nic_fw_entry_t
   rocm.bindings.amdsmi.amdsmi_nic_fw_info_t
   rocm.bindings.amdsmi.amdsmi_nic_port_t
   rocm.bindings.amdsmi.amdsmi_nic_port_info_t
   rocm.bindings.amdsmi.amdsmi_nic_driver_info_t
   rocm.bindings.amdsmi.amdsmi_nic_rdma_port_info_t
   rocm.bindings.amdsmi.amdsmi_nic_rdma_dev_info_t
   rocm.bindings.amdsmi.amdsmi_nic_rdma_devices_info_t
   rocm.bindings.amdsmi.amdsmi_fabric_telemetry_category_t
   rocm.bindings.amdsmi.amdsmi_fabric_telemetry_category_mask_t
   rocm.bindings.amdsmi.amdsmi_fabric_telemetry_item_t
   rocm.bindings.amdsmi.amdsmi_fabric_label_t
   rocm.bindings.amdsmi.amdsmi_fabric_telemetry_instance_t
   rocm.bindings.amdsmi.amdsmi_fabric_telemetry_dataset_t
   rocm.bindings.amdsmi.amdsmi_fabric_telemetry_t
   rocm.bindings.amdsmi.amdsmi_fabric_size_constants_t
   rocm.bindings.amdsmi.amdsmi_fabric_type_t
   rocm.bindings.amdsmi.amdsmi_fabric_npa_address_mode_t
   rocm.bindings.amdsmi.amdsmi_fabric_accelerator_vpod_state_t
   rocm.bindings.amdsmi.amdsmi_fabric_info_v1_t
   rocm.bindings.amdsmi.amdsmi_fabric_info_t_fabric_info_
   rocm.bindings.amdsmi.amdsmi_fabric_info_t
   rocm.bindings.amdsmi.amdsmi_cper_guid_t
   rocm.bindings.amdsmi.amdsmi_cper_timestamp_t
   rocm.bindings.amdsmi.amdsmi_cper_valid_bits_t_valid_bits_
   rocm.bindings.amdsmi.amdsmi_cper_valid_bits_t
   rocm.bindings.amdsmi.amdsmi_cper_hdr_t
   rocm.bindings.amdsmi.amdsmi_uma_carveout_option_t
   rocm.bindings.amdsmi.amdsmi_uma_carveout_info_t
   rocm.bindings.amdsmi.amdsmi_ttm_info_t


Functions
---------

.. autoapisummary::

   rocm.bindings.amdsmi.has_symbol
   rocm.bindings.amdsmi.amdsmi_init
   rocm.bindings.amdsmi.amdsmi_shut_down
   rocm.bindings.amdsmi.amdsmi_get_socket_handles
   rocm.bindings.amdsmi.amdsmi_get_socket_info
   rocm.bindings.amdsmi.amdsmi_get_processor_handles
   rocm.bindings.amdsmi.amdsmi_get_node_handle
   rocm.bindings.amdsmi.amdsmi_get_processor_type
   rocm.bindings.amdsmi.amdsmi_get_processor_info
   rocm.bindings.amdsmi.amdsmi_get_processor_count_from_handles
   rocm.bindings.amdsmi.amdsmi_get_processor_handles_by_type
   rocm.bindings.amdsmi.amdsmi_get_processor_handle_from_bdf
   rocm.bindings.amdsmi.amdsmi_get_gpu_device_bdf
   rocm.bindings.amdsmi.amdsmi_get_gpu_device_uuid
   rocm.bindings.amdsmi.amdsmi_get_gpu_enumeration_info
   rocm.bindings.amdsmi.amdsmi_get_cpu_affinity_with_scope
   rocm.bindings.amdsmi.amdsmi_get_gpu_virtualization_mode
   rocm.bindings.amdsmi.amdsmi_get_nic_processor_handles
   rocm.bindings.amdsmi.amdsmi_get_nic_device_bdf
   rocm.bindings.amdsmi.amdsmi_get_gpu_id
   rocm.bindings.amdsmi.amdsmi_get_gpu_revision
   rocm.bindings.amdsmi.amdsmi_get_gpu_vendor_name
   rocm.bindings.amdsmi.amdsmi_get_gpu_vram_vendor
   rocm.bindings.amdsmi.amdsmi_get_gpu_subsystem_id
   rocm.bindings.amdsmi.amdsmi_get_gpu_subsystem_name
   rocm.bindings.amdsmi.amdsmi_get_gpu_pci_bandwidth
   rocm.bindings.amdsmi.amdsmi_get_gpu_bdf_id
   rocm.bindings.amdsmi.amdsmi_get_gpu_topo_numa_affinity
   rocm.bindings.amdsmi.amdsmi_get_gpu_pci_throughput
   rocm.bindings.amdsmi.amdsmi_get_gpu_pci_replay_counter
   rocm.bindings.amdsmi.amdsmi_set_gpu_pci_bandwidth
   rocm.bindings.amdsmi.amdsmi_get_energy_count
   rocm.bindings.amdsmi.amdsmi_set_power_cap
   rocm.bindings.amdsmi.amdsmi_set_gpu_power_profile
   rocm.bindings.amdsmi.amdsmi_get_supported_power_cap
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_power
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_power_cap
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_power_cap_max
   rocm.bindings.amdsmi.amdsmi_get_cpu_pwr_svi_telemetry_all_rails
   rocm.bindings.amdsmi.amdsmi_set_cpu_socket_power_cap
   rocm.bindings.amdsmi.amdsmi_set_cpu_pwr_efficiency_mode
   rocm.bindings.amdsmi.amdsmi_get_cpu_pwr_efficiency_mode
   rocm.bindings.amdsmi.amdsmi_get_cpu_core_ccd_power
   rocm.bindings.amdsmi.amdsmi_get_gpu_memory_total
   rocm.bindings.amdsmi.amdsmi_get_gpu_memory_usage
   rocm.bindings.amdsmi.amdsmi_get_gpu_bad_page_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_bad_page_threshold
   rocm.bindings.amdsmi.amdsmi_gpu_validate_ras_eeprom
   rocm.bindings.amdsmi.amdsmi_get_gpu_ras_block_features_enabled
   rocm.bindings.amdsmi.amdsmi_get_gpu_memory_reserved_pages
   rocm.bindings.amdsmi.amdsmi_get_gpu_fan_rpms
   rocm.bindings.amdsmi.amdsmi_get_gpu_fan_speed
   rocm.bindings.amdsmi.amdsmi_get_gpu_fan_speed_max
   rocm.bindings.amdsmi.amdsmi_get_gpu_cache_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_volt_metric
   rocm.bindings.amdsmi.amdsmi_reset_gpu_fan
   rocm.bindings.amdsmi.amdsmi_set_gpu_fan_speed
   rocm.bindings.amdsmi.amdsmi_get_gpu_busy_percent
   rocm.bindings.amdsmi.amdsmi_get_vcn_busy_percent
   rocm.bindings.amdsmi.amdsmi_get_utilization_count
   rocm.bindings.amdsmi.amdsmi_get_gpu_perf_level
   rocm.bindings.amdsmi.amdsmi_set_gpu_perf_determinism_mode
   rocm.bindings.amdsmi.amdsmi_get_gpu_overdrive_level
   rocm.bindings.amdsmi.amdsmi_get_gpu_mem_overdrive_level
   rocm.bindings.amdsmi.amdsmi_get_clk_freq
   rocm.bindings.amdsmi.amdsmi_reset_gpu
   rocm.bindings.amdsmi.amdsmi_get_gpu_od_volt_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_metrics_header_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_metrics_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_partition_metrics_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_pm_metrics_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_reg_table_info
   rocm.bindings.amdsmi.amdsmi_set_gpu_clk_limit
   rocm.bindings.amdsmi.amdsmi_set_gpu_od_clk_info
   rocm.bindings.amdsmi.amdsmi_set_gpu_od_volt_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_od_volt_curve_regions
   rocm.bindings.amdsmi.amdsmi_get_gpu_power_profile_presets
   rocm.bindings.amdsmi.amdsmi_set_gpu_perf_level
   rocm.bindings.amdsmi.amdsmi_set_gpu_overdrive_level
   rocm.bindings.amdsmi.amdsmi_set_clk_freq
   rocm.bindings.amdsmi.amdsmi_get_soc_pstate
   rocm.bindings.amdsmi.amdsmi_set_soc_pstate
   rocm.bindings.amdsmi.amdsmi_get_xgmi_plpd
   rocm.bindings.amdsmi.amdsmi_set_xgmi_plpd
   rocm.bindings.amdsmi.amdsmi_get_gpu_process_isolation
   rocm.bindings.amdsmi.amdsmi_set_gpu_process_isolation
   rocm.bindings.amdsmi.amdsmi_clean_gpu_local_data
   rocm.bindings.amdsmi.amdsmi_alloc_fabric_telemetry
   rocm.bindings.amdsmi.amdsmi_get_fabric_telemetry_data
   rocm.bindings.amdsmi.amdsmi_fabric_telem_id_to_string
   rocm.bindings.amdsmi.amdsmi_free_fabric_telemetry
   rocm.bindings.amdsmi.amdsmi_get_gpu_fabric_info
   rocm.bindings.amdsmi.amdsmi_get_lib_version
   rocm.bindings.amdsmi.amdsmi_get_gpu_ecc_count
   rocm.bindings.amdsmi.amdsmi_get_gpu_ecc_enabled
   rocm.bindings.amdsmi.amdsmi_get_gpu_total_ecc_count
   rocm.bindings.amdsmi.amdsmi_get_afids_from_cper
   rocm.bindings.amdsmi.amdsmi_get_gpu_ras_feature_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_cper_entries
   rocm.bindings.amdsmi.amdsmi_get_gpu_ecc_status
   rocm.bindings.amdsmi.amdsmi_status_code_to_string
   rocm.bindings.amdsmi.amdsmi_gpu_counter_group_supported
   rocm.bindings.amdsmi.amdsmi_gpu_create_counter
   rocm.bindings.amdsmi.amdsmi_gpu_destroy_counter
   rocm.bindings.amdsmi.amdsmi_gpu_control_counter
   rocm.bindings.amdsmi.amdsmi_gpu_read_counter
   rocm.bindings.amdsmi.amdsmi_get_gpu_available_counters
   rocm.bindings.amdsmi.amdsmi_get_gpu_compute_process_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_compute_process_info_by_pid
   rocm.bindings.amdsmi.amdsmi_get_gpu_compute_process_gpus
   rocm.bindings.amdsmi.amdsmi_gpu_xgmi_error_status
   rocm.bindings.amdsmi.amdsmi_reset_gpu_xgmi_error
   rocm.bindings.amdsmi.amdsmi_get_xgmi_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_xgmi_link_status
   rocm.bindings.amdsmi.amdsmi_get_link_metrics
   rocm.bindings.amdsmi.amdsmi_topo_get_numa_node_number
   rocm.bindings.amdsmi.amdsmi_topo_get_link_weight
   rocm.bindings.amdsmi.amdsmi_get_minmax_bandwidth_between_processors
   rocm.bindings.amdsmi.amdsmi_topo_get_link_type
   rocm.bindings.amdsmi.amdsmi_get_link_topology_nearest
   rocm.bindings.amdsmi.amdsmi_is_P2P_accessible
   rocm.bindings.amdsmi.amdsmi_topo_get_p2p_status
   rocm.bindings.amdsmi.amdsmi_get_gpu_compute_partition
   rocm.bindings.amdsmi.amdsmi_set_gpu_compute_partition
   rocm.bindings.amdsmi.amdsmi_get_gpu_compute_partition_mem_alloc_mode
   rocm.bindings.amdsmi.amdsmi_get_gpu_accelerator_partition_mem_alloc_mode
   rocm.bindings.amdsmi.amdsmi_set_gpu_compute_partition_mem_alloc_mode
   rocm.bindings.amdsmi.amdsmi_set_gpu_accelerator_partition_mem_alloc_mode
   rocm.bindings.amdsmi.amdsmi_get_gpu_memory_partition
   rocm.bindings.amdsmi.amdsmi_set_gpu_memory_partition
   rocm.bindings.amdsmi.amdsmi_get_gpu_memory_partition_config
   rocm.bindings.amdsmi.amdsmi_set_gpu_memory_partition_mode
   rocm.bindings.amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config
   rocm.bindings.amdsmi.amdsmi_get_gpu_accelerator_partition_profile
   rocm.bindings.amdsmi.amdsmi_set_gpu_accelerator_partition_profile
   rocm.bindings.amdsmi.amdsmi_init_gpu_event_notification
   rocm.bindings.amdsmi.amdsmi_set_gpu_event_notification_mask
   rocm.bindings.amdsmi.amdsmi_get_gpu_event_notification
   rocm.bindings.amdsmi.amdsmi_stop_gpu_event_notification
   rocm.bindings.amdsmi.amdsmi_get_gpu_driver_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_asic_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_kfd_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_vram_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_board_info
   rocm.bindings.amdsmi.amdsmi_get_power_cap_info
   rocm.bindings.amdsmi.amdsmi_get_pcie_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_xcd_counter
   rocm.bindings.amdsmi.amdsmi_get_npm_info
   rocm.bindings.amdsmi.amdsmi_get_fw_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_vbios_info
   rocm.bindings.amdsmi.amdsmi_get_temp_metric
   rocm.bindings.amdsmi.amdsmi_get_gpu_activity
   rocm.bindings.amdsmi.amdsmi_get_power_info
   rocm.bindings.amdsmi.amdsmi_is_gpu_power_management_enabled
   rocm.bindings.amdsmi.amdsmi_get_clock_info
   rocm.bindings.amdsmi.amdsmi_get_gpu_vram_usage
   rocm.bindings.amdsmi.amdsmi_get_violation_status
   rocm.bindings.amdsmi.amdsmi_get_gpu_process_list
   rocm.bindings.amdsmi.amdsmi_get_gpu_process_list_by_pid
   rocm.bindings.amdsmi.amdsmi_get_gpu_ptl_state
   rocm.bindings.amdsmi.amdsmi_set_gpu_ptl_state
   rocm.bindings.amdsmi.amdsmi_get_gpu_ptl_formats
   rocm.bindings.amdsmi.amdsmi_set_gpu_ptl_formats
   rocm.bindings.amdsmi.amdsmi_get_cpu_handles
   rocm.bindings.amdsmi.amdsmi_get_cpucore_handles
   rocm.bindings.amdsmi.amdsmi_get_cpu_core_energy
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_energy
   rocm.bindings.amdsmi.amdsmi_get_threads_per_core
   rocm.bindings.amdsmi.amdsmi_get_cpu_hsmp_driver_version
   rocm.bindings.amdsmi.amdsmi_get_cpu_smu_fw_version
   rocm.bindings.amdsmi.amdsmi_get_cpu_hsmp_proto_ver
   rocm.bindings.amdsmi.amdsmi_get_cpu_prochot_status
   rocm.bindings.amdsmi.amdsmi_get_cpu_fclk_mclk
   rocm.bindings.amdsmi.amdsmi_get_cpu_cclk_limit
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_current_active_freq_limit
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_freq_range
   rocm.bindings.amdsmi.amdsmi_get_cpu_core_current_freq_limit
   rocm.bindings.amdsmi.amdsmi_set_cpu_rail_isofreq_policy
   rocm.bindings.amdsmi.amdsmi_get_cpu_rail_isofreq_policy
   rocm.bindings.amdsmi.amdsmi_set_cpu_dfc_ctrl
   rocm.bindings.amdsmi.amdsmi_get_cpu_dfc_ctrl
   rocm.bindings.amdsmi.amdsmi_get_cpu_core_boostlimit
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_c0_residency
   rocm.bindings.amdsmi.amdsmi_set_cpu_core_boostlimit
   rocm.bindings.amdsmi.amdsmi_set_cpu_socket_boostlimit
   rocm.bindings.amdsmi.amdsmi_get_cpu_core_floor_freq_limit
   rocm.bindings.amdsmi.amdsmi_get_cpu_floor_freq_limit
   rocm.bindings.amdsmi.amdsmi_get_cpu_core_eff_floor_freq_limit
   rocm.bindings.amdsmi.amdsmi_get_cpu_eff_floor_freq_limit
   rocm.bindings.amdsmi.amdsmi_set_cpu_core_floor_freq_limit
   rocm.bindings.amdsmi.amdsmi_set_cpu_floor_freq_limit
   rocm.bindings.amdsmi.amdsmi_set_cpu_msr_floor_freq_limit
   rocm.bindings.amdsmi.amdsmi_set_cpu_core_msr_floor_freq_limit
   rocm.bindings.amdsmi.amdsmi_get_cpu_freq_range
   rocm.bindings.amdsmi.amdsmi_set_cpu_sdps_limit
   rocm.bindings.amdsmi.amdsmi_get_cpu_sdps_limit
   rocm.bindings.amdsmi.amdsmi_get_cpu_ddr_bw
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_temperature
   rocm.bindings.amdsmi.amdsmi_get_cpu_tdelta
   rocm.bindings.amdsmi.amdsmi_get_cpu_svi3_vr_controller_temp
   rocm.bindings.amdsmi.amdsmi_get_cpu_dimm_temp_range_and_refresh_rate
   rocm.bindings.amdsmi.amdsmi_get_cpu_dimm_power_consumption
   rocm.bindings.amdsmi.amdsmi_get_cpu_dimm_thermal_sensor
   rocm.bindings.amdsmi.amdsmi_get_cpu_dimm_sb_reg
   rocm.bindings.amdsmi.amdsmi_set_cpu_dimm_sb_reg
   rocm.bindings.amdsmi.amdsmi_set_cpu_xgmi_width
   rocm.bindings.amdsmi.amdsmi_set_cpu_gmi3_link_width_range
   rocm.bindings.amdsmi.amdsmi_cpu_apb_enable
   rocm.bindings.amdsmi.amdsmi_cpu_apb_disable
   rocm.bindings.amdsmi.amdsmi_set_cpu_socket_lclk_dpm_level
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_lclk_dpm_level
   rocm.bindings.amdsmi.amdsmi_set_cpu_pcie_link_rate
   rocm.bindings.amdsmi.amdsmi_set_cpu_df_pstate_range
   rocm.bindings.amdsmi.amdsmi_set_cpu_xgmi_pstate_range
   rocm.bindings.amdsmi.amdsmi_get_cpu_xgmi_pstate_range
   rocm.bindings.amdsmi.amdsmi_get_cpu_pc6_enable
   rocm.bindings.amdsmi.amdsmi_set_cpu_pc6_enable
   rocm.bindings.amdsmi.amdsmi_get_cpu_cc6_enable
   rocm.bindings.amdsmi.amdsmi_set_cpu_cc6_enable
   rocm.bindings.amdsmi.amdsmi_get_cpu_current_io_bandwidth
   rocm.bindings.amdsmi.amdsmi_get_cpu_current_xgmi_bw
   rocm.bindings.amdsmi.amdsmi_get_hsmp_metrics_table_version
   rocm.bindings.amdsmi.amdsmi_get_hsmp_metrics_table
   rocm.bindings.amdsmi.amdsmi_first_online_core_on_cpu_socket
   rocm.bindings.amdsmi.amdsmi_get_cpu_family
   rocm.bindings.amdsmi.amdsmi_get_cpu_model
   rocm.bindings.amdsmi.amdsmi_get_cpu_model_name
   rocm.bindings.amdsmi.amdsmi_get_esmi_err_msg
   rocm.bindings.amdsmi.amdsmi_get_cpu_cores_per_socket
   rocm.bindings.amdsmi.amdsmi_get_cpu_socket_count
   rocm.bindings.amdsmi.amdsmi_get_cpu_enabled_commands
   rocm.bindings.amdsmi.amdsmi_get_nic_driver_info
   rocm.bindings.amdsmi.amdsmi_get_nic_asic_info
   rocm.bindings.amdsmi.amdsmi_get_nic_bus_info
   rocm.bindings.amdsmi.amdsmi_get_nic_numa_info
   rocm.bindings.amdsmi.amdsmi_get_nic_port_info
   rocm.bindings.amdsmi.amdsmi_get_nic_rdma_dev_info
   rocm.bindings.amdsmi.amdsmi_get_nic_rdma_port_statistics
   rocm.bindings.amdsmi.amdsmi_get_nic_fw_info
   rocm.bindings.amdsmi.amdsmi_get_nic_port_statistics
   rocm.bindings.amdsmi.amdsmi_get_nic_vendor_statistics
   rocm.bindings.amdsmi.amdsmi_get_gpu_uma_carveout_info
   rocm.bindings.amdsmi.amdsmi_set_gpu_uma_carveout
   rocm.bindings.amdsmi.amdsmi_get_ttm_info
   rocm.bindings.amdsmi.amdsmi_set_ttm_pages_limit
   rocm.bindings.amdsmi.amdsmi_reset_ttm_pages_limit


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: AMDSMI_MAX_MM_IP_COUNT
   :type:  Any

.. py:data:: AMDSMI_MAX_STRING_LENGTH
   :type:  Any

.. py:data:: AMDSMI_MAX_DEVICES
   :type:  Any

.. py:data:: AMDSMI_MAX_CACHE_TYPES
   :type:  Any

.. py:data:: AMDSMI_MAX_ACCELERATOR_PROFILE
   :type:  Any

.. py:data:: AMDSMI_MAX_CP_PROFILE_RESOURCES
   :type:  Any

.. py:data:: AMDSMI_MAX_ACCELERATOR_PARTITIONS
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_NUMA_NODES
   :type:  Any

.. py:data:: AMDSMI_GPU_UUID_SIZE
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK
   :type:  Any

.. py:data:: AMDSMI_MAX_CONTAINER_TYPE
   :type:  Any

.. py:data:: AMDSMI_NUM_HBM_INSTANCES
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_VCN
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_CLKS
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_XGMI_LINKS
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_GFX_CLKS
   :type:  Any

.. py:data:: AMDSMI_MAX_AID
   :type:  Any

.. py:data:: AMDSMI_MAX_ENGINES
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_JPEG
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_JPEG_ENG_V1
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_XCC
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_XCP
   :type:  Any

.. py:data:: AMDSMI_APU_MAX_CORES
   :type:  Any

.. py:data:: AMDSMI_APU_V24_CORES
   :type:  Any

.. py:data:: AMDSMI_APU_MAX_L3
   :type:  Any

.. py:data:: AMDSMI_APU_MAX_IPU
   :type:  Any

.. py:data:: AMDSMI_MAX_NUMBER_OF_AFIDS_PER_RECORD
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_HBM_STACKS
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_AID
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_MID
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_CLKS_PER_AID
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_CLKS_PER_MID
   :type:  Any

.. py:data:: AMDSMI_TIME_FORMAT
   :type:  Any

.. py:data:: AMDSMI_DATE_FORMAT
   :type:  Any

.. py:data:: AMDSMI_LIB_VERSION_MAJOR
   :type:  Any

.. py:data:: AMDSMI_LIB_VERSION_MINOR
   :type:  Any

.. py:data:: AMDSMI_LIB_VERSION_RELEASE
   :type:  Any

.. py:data:: AMDSMI_LIB_VERSION_STRING
   :type:  Any

.. py:data:: AMDSMI_MAX_DRIVER_INFO_RSVD
   :type:  Any

.. py:data:: AMDSMI_MAX_UUID_ELEMENTS
   :type:  Any

.. py:data:: AMDSMI_MAX_SPD_DIMM_ADDRESS
   :type:  Any

.. py:data:: AMDSMI_MAX_SPD_LID
   :type:  Any

.. py:data:: AMDSMI_MAX_SPD_REG_OFFSET
   :type:  Any

.. py:data:: AMDSMI_MAX_SPD_REG_SPACE
   :type:  Any

.. py:data:: AMDSMI_MAX_SPD_WRITE_DATA
   :type:  Any

.. py:data:: AMDSMI_MAX_SVI3_RAIL_INDEX
   :type:  Any

.. py:data:: AMDSMI_MAX_SVI3_RAIL_SELECTION
   :type:  Any

.. py:data:: AMDSMI_POWER_EFFICIENCY_MODE_4
   :type:  Any

.. py:data:: AMDSMI_POWER_EFFICIENCY_MODE_5
   :type:  Any

.. py:data:: AMDSMI_MAX_POWER_EFFICIENCY_UTIL
   :type:  Any

.. py:data:: AMDSMI_MAX_POWER_EFFICIENCY_PPTLIMIT
   :type:  Any

.. py:data:: AMDSMI_RAIL_INDEX_NONE
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_FREQUENCIES
   :type:  Any

.. py:data:: AMDSMI_MAX_FAN_SPEED
   :type:  Any

.. py:data:: AMDSMI_NUM_VOLTAGE_CURVE_POINTS
   :type:  Any

.. py:data:: AMDSMI_MAX_UTILIZATION_VALUES
   :type:  Any

.. py:data:: AMDSMI_MAX_NUM_PM_POLICIES
   :type:  Any

.. py:data:: AMDSMI_MAX_NIC_PORTS
   :type:  Any

.. py:data:: AMDSMI_MAX_NIC_RDMA_DEV
   :type:  Any

.. py:data:: AMDSMI_MAX_NIC_FW
   :type:  Any

.. py:data:: AMDSMI_FABRIC_LABEL_MAX_LENGTH
   :type:  Any

.. py:data:: AMDSMI_FABRIC_PPOD_ID_SIZE
   :type:  Any

.. py:data:: AMDSMI_MAX_CARVEOUT_OPTIONS
   :type:  Any

.. py:class:: timespec(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: tv_sec
      :type:  Any


   .. py:attribute:: tv_nsec
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_init_flags_t

   Bases: :py:obj:`enum.IntEnum`


   Initialization flags

   Initialization flags may be OR'd together and passed to ::amdsmi_init().


   .. py:attribute:: AMDSMI_INIT_ALL_PROCESSORS
      :type:  int


   .. py:attribute:: AMDSMI_INIT_AMD_CPUS
      :type:  int


   .. py:attribute:: AMDSMI_INIT_AMD_GPUS
      :type:  int


   .. py:attribute:: AMDSMI_INIT_NON_AMD_CPUS
      :type:  int


   .. py:attribute:: AMDSMI_INIT_NON_AMD_GPUS
      :type:  int


   .. py:attribute:: AMDSMI_INIT_AMD_APUS
      :type:  int


   .. py:attribute:: AMDSMI_INIT_AMD_NICS
      :type:  int


.. py:class:: amdsmi_mm_ip_t

   Bases: :py:obj:`enum.IntEnum`


   GPU Capability info
       


   .. py:attribute:: AMDSMI_MM_UVD
      :type:  int


   .. py:attribute:: AMDSMI_MM_VCE
      :type:  int


   .. py:attribute:: AMDSMI_MM_VCN
      :type:  int


   .. py:attribute:: AMDSMI_MM__MAX
      :type:  int


.. py:class:: amdsmi_container_types_t

   Bases: :py:obj:`enum.IntEnum`


   Container
       


   .. py:attribute:: AMDSMI_CONTAINER_LXC
      :type:  int


   .. py:attribute:: AMDSMI_CONTAINER_DOCKER
      :type:  int


.. py:class:: amdsmi_hsmp_driver_version_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds HSMP Driver version information.
       


   .. py:attribute:: major
      :type:  Any


   .. py:attribute:: minor
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_processor_type_t

   Bases: :py:obj:`enum.IntEnum`


   Processor types detectable by AMD SMI
       


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_UNKNOWN
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_AMD_GPU
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_AMD_CPU
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_AMD_APU
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_AMD_NIC
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_BRCM_NIC
      :type:  int


   .. py:attribute:: AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH
      :type:  int


.. py:data:: processor_type_t

.. py:class:: amdsmi_status_t

   Bases: :py:obj:`enum.IntEnum`


   Error codes returned by amdsmi functions

   Please avoid status codes that are multiples of 256 (256, 512, etc..)
   Return values in the shell get modulo 256 applied, meaning any multiple of 256 ends up as 0


   .. py:attribute:: AMDSMI_STATUS_SUCCESS
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_INVAL
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NOT_SUPPORTED
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NOT_YET_IMPLEMENTED
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_FAIL_LOAD_MODULE
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_FAIL_LOAD_SYMBOL
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_DRM_ERROR
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_API_FAILED
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_TIMEOUT
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_RETRY
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_PERM
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_INTERRUPT
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_IO
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_ADDRESS_FAULT
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_FILE_ERROR
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_OUT_OF_RESOURCES
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_INTERNAL_EXCEPTION
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_INIT_ERROR
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_REFCOUNT_OVERFLOW
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_DIRECTORY_NOT_FOUND
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_IPC_ERROR
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_BUSY
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NOT_FOUND
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NOT_INIT
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_SLOT
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_DRIVER_NOT_LOADED
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_MORE_DATA
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_DATA
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_INSUFFICIENT_SIZE
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_UNEXPECTED_SIZE
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_UNEXPECTED_DATA
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NON_AMD_CPU
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_ENERGY_DRV
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_MSR_DRV
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_HSMP_DRV
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_HSMP_SUP
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_HSMP_MSG_SUP
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_HSMP_TIMEOUT
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_NO_DRV
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_FILE_NOT_FOUND
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_ARG_PTR_NULL
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_AMDGPU_RESTART_ERR
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_SETTING_UNAVAILABLE
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_CORRUPTED_EEPROM
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_MAP_ERROR
      :type:  int


   .. py:attribute:: AMDSMI_STATUS_UNKNOWN_ERROR
      :type:  int


.. py:class:: amdsmi_clk_type_t

   Bases: :py:obj:`enum.IntEnum`


   Clock types
       


   .. py:attribute:: AMDSMI_CLK_TYPE_SYS
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_GFX
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_DF
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_DCEF
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_SOC
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_MEM
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_PCIE
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_VCLK0
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_VCLK1
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_DCLK0
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE_DCLK1
      :type:  int


   .. py:attribute:: AMDSMI_CLK_TYPE__MAX
      :type:  int


.. py:class:: amdsmi_accelerator_partition_type_t

   Bases: :py:obj:`enum.IntEnum`


   Accelerator Partition
       


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_INVALID
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_SPX
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_DPX
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_TPX
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_QPX
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_CPX
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_MAX
      :type:  int


.. py:class:: amdsmi_accelerator_partition_resource_type_t

   Bases: :py:obj:`enum.IntEnum`


   Accelerator Partition Resource Types
       


   .. py:attribute:: AMDSMI_ACCELERATOR_XCC
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_ENCODER
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_DECODER
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_DMA
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_JPEG
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_MAX
      :type:  int


.. py:class:: amdsmi_compute_partition_type_t

   Bases: :py:obj:`enum.IntEnum`


   Compute Partition. This enum is used to identify
   various compute partitioning settings.

   Deprecated:
       This enum is slated for removal in a future ROCm release;
       use amdsmi_accelerator_partition_type_t instead


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_INVALID
      :type:  int


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_SPX
      :type:  int


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_DPX
      :type:  int


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_TPX
      :type:  int


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_QPX
      :type:  int


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_CPX
      :type:  int


.. py:class:: amdsmi_compute_partition_mem_alloc_mode_t

   Bases: :py:obj:`enum.IntEnum`


   Compute Partition Memory Allocation Mode. Controls how GPU memory
   is allocated across XCPs within a memory partition.

   Deprecated:
       This enum is slated for removal in a future ROCm release;
       use amdsmi_accelerator_partition_mem_alloc_mode_t instead


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_INVALID
      :type:  int


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_CAPPING
      :type:  int


   .. py:attribute:: AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_ALL
      :type:  int


.. py:class:: amdsmi_accelerator_partition_mem_alloc_mode_t

   Bases: :py:obj:`enum.IntEnum`


   Accelerator Partition Memory Allocation Mode. Controls how GPU memory
   is allocated across XCPs within a memory partition.


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_INVALID
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING
      :type:  int


   .. py:attribute:: AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_ALL
      :type:  int


.. py:class:: amdsmi_memory_partition_type_t

   Bases: :py:obj:`enum.IntEnum`


   Memory Partitions
       


   .. py:attribute:: AMDSMI_MEMORY_PARTITION_UNKNOWN
      :type:  int


   .. py:attribute:: AMDSMI_MEMORY_PARTITION_NPS1
      :type:  int


   .. py:attribute:: AMDSMI_MEMORY_PARTITION_NPS2
      :type:  int


   .. py:attribute:: AMDSMI_MEMORY_PARTITION_NPS4
      :type:  int


   .. py:attribute:: AMDSMI_MEMORY_PARTITION_NPS8
      :type:  int


.. py:class:: amdsmi_temperature_type_t

   Bases: :py:obj:`enum.IntEnum`


   This enumeration is used to indicate from which part of the processor a
   temperature reading should be obtained.


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_EDGE
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_HOTSPOT
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_JUNCTION
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_VRAM
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_HBM_0
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_HBM_1
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_HBM_2
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_HBM_3
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_PLX
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_RETIMER_X
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC_2
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_LAST
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD0
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD1
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD2
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD3
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_A
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_C
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_A
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_C
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_085_HBM
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_B
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_D
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_USR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_E32
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_04_HBM_B
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_04_HBM_D
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_HBM_B
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_HBM_D
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_GTA_A
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_GTA_C
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075_GTA_A
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075_GTA_C
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_075_UCIE
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAA
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAM_A
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_065_UCIEAM_C
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDAN_075
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_LAST
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_GPUBOARD_LAST
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FRONT
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_BACK
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM7
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_IBC
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_UFPGA
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM1
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_HSC
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_2_3_HSC
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_HSC
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_6_7_HSC
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_0V72_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_3V3_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC_HSC
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST
      :type:  int


   .. py:attribute:: AMDSMI_TEMPERATURE_TYPE__MAX
      :type:  int


.. py:class:: amdsmi_fw_block_t

   Bases: :py:obj:`enum.IntEnum`


   The values of this enum are used to identify the various firmware
   blocks.


   .. py:attribute:: AMDSMI_FW_ID_SMU
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_CE
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_PFP
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_ME
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_MEC_JT1
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_MEC_JT2
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_MEC1
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_MEC2
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA0
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA1
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA2
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA3
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA4
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA5
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA6
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA7
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_VCN
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_UVD
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_VCE
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_ISP
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_DMCU_ERAM
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_DMCU_ISR
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC_V
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_MMSCH
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_SYSDRV
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_SOSDRV
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_TOC
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_KEYDB
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_DFC
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_SPL
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_DRV_CAP
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_MC
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_BL
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_PM4
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC_P
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SEC_POLICY_STAGE2
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_REG_ACCESS_WHITELIST
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_IMU_DRAM
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_IMU_IRAM
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA_TH0
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_SDMA_TH1
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_CP_MES
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_MES_KIQ
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_MES_STACK
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_MES_THREAD1
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_MES_THREAD1_STACK
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLX6
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLX6_DRAM_BOOT
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_ME
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_ME_P0_DATA
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_ME_P1_DATA
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_PFP
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_PFP_P0_DATA
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_PFP_P1_DATA
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_MEC
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_MEC_P0_DATA
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_MEC_P1_DATA
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_MEC_P2_DATA
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RS64_MEC_P3_DATA
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PPTABLE
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_SOC
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_DBG
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PSP_INTF
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLX6_CORE1
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLCV_LX7
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_ASD
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_TA_RAS
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_TA_XGMI
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC_SRLG
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_RLC_SRLS
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PM
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_DMCU
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID_PLDM_BUNDLE
      :type:  int


   .. py:attribute:: AMDSMI_FW_ID__MAX
      :type:  int


.. py:class:: amdsmi_vram_type_t

   Bases: :py:obj:`enum.IntEnum`


   vRam Types. This enum is used to identify various VRam types.
       


   .. py:attribute:: AMDSMI_VRAM_TYPE_UNKNOWN
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_HBM
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_HBM2
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_HBM2E
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_HBM3
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_HBM3E
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_DDR2
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_DDR3
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_DDR4
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_DDR5
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_GDDR1
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_GDDR2
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_GDDR3
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_GDDR4
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_GDDR5
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_GDDR6
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_GDDR7
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_LPDDR4
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE_LPDDR5
      :type:  int


   .. py:attribute:: AMDSMI_VRAM_TYPE__MAX
      :type:  int


.. py:class:: amdsmi_range_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure represents a range (e.g., frequencies or voltages).
       


   .. py:attribute:: lower_bound
      :type:  Any


   .. py:attribute:: upper_bound
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_xgmi_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   XGMI Information
       


   .. py:attribute:: xgmi_lanes
      :type:  Any


   .. py:attribute:: xgmi_hive_id
      :type:  Any


   .. py:attribute:: xgmi_node_id
      :type:  Any


   .. py:attribute:: index
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_vram_usage_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   VRam Usage
       


   .. py:attribute:: vram_total
      :type:  Any


   .. py:attribute:: vram_used
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_violation_status_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure hold violation status information.
   Note: for MI3x asics and higher, older ASICs will show unsupported.


   .. py:attribute:: reference_timestamp
      :type:  Any


   .. py:attribute:: violation_timestamp
      :type:  Any


   .. py:attribute:: acc_counter
      :type:  Any


   .. py:attribute:: acc_prochot_thrm
      :type:  Any


   .. py:attribute:: acc_ppt_pwr
      :type:  Any


   .. py:attribute:: acc_socket_thrm
      :type:  Any


   .. py:attribute:: acc_vr_thrm
      :type:  Any


   .. py:attribute:: acc_hbm_thrm
      :type:  Any


   .. py:attribute:: acc_gfx_clk_below_host_limit
      :type:  Any


   .. py:attribute:: per_prochot_thrm
      :type:  Any


   .. py:attribute:: per_ppt_pwr
      :type:  Any


   .. py:attribute:: per_socket_thrm
      :type:  Any


   .. py:attribute:: per_vr_thrm
      :type:  Any


   .. py:attribute:: per_hbm_thrm
      :type:  Any


   .. py:attribute:: per_gfx_clk_below_host_limit
      :type:  Any


   .. py:attribute:: active_prochot_thrm
      :type:  Any


   .. py:attribute:: active_ppt_pwr
      :type:  Any


   .. py:attribute:: active_socket_thrm
      :type:  Any


   .. py:attribute:: active_vr_thrm
      :type:  Any


   .. py:attribute:: active_hbm_thrm
      :type:  Any


   .. py:attribute:: active_gfx_clk_below_host_limit
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_frequency_range_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Frequency Range
       


   .. py:attribute:: supported_freq_range
      :type:  Any


   .. py:attribute:: current_freq_range
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_bdf_t_bdf_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: function_number
      :type:  Any


   .. py:attribute:: device_number
      :type:  Any


   .. py:attribute:: bus_number
      :type:  Any


   .. py:attribute:: domain_number
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_bdf_t_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: function_number
      :type:  Any


   .. py:attribute:: device_number
      :type:  Any


   .. py:attribute:: bus_number
      :type:  Any


   .. py:attribute:: domain_number
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_bdf_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   bdf types
       


   .. py:attribute:: bdf
      :type:  Any


   .. py:attribute:: as_uint
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_enumeration_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Structure holds enumeration information
       


   .. py:attribute:: drm_render
      :type:  Any


   .. py:attribute:: drm_card
      :type:  Any


   .. py:attribute:: hsa_id
      :type:  Any


   .. py:attribute:: hip_id
      :type:  Any


   .. py:attribute:: hip_uuid
      :type:  Any


   .. py:attribute:: oam_id
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_card_form_factor_t

   Bases: :py:obj:`enum.IntEnum`


   Card Form Factor
       


   .. py:attribute:: AMDSMI_CARD_FORM_FACTOR_PCIE
      :type:  int


   .. py:attribute:: AMDSMI_CARD_FORM_FACTOR_OAM
      :type:  int


   .. py:attribute:: AMDSMI_CARD_FORM_FACTOR_CEM
      :type:  int


   .. py:attribute:: AMDSMI_CARD_FORM_FACTOR_UNKNOWN
      :type:  int


.. py:class:: amdsmi_pcie_info_t_pcie_static_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: max_pcie_width
      :type:  Any


   .. py:attribute:: max_pcie_speed
      :type:  Any


   .. py:attribute:: pcie_interface_version
      :type:  Any


   .. py:attribute:: slot_type
      :type:  Any


   .. py:attribute:: max_pcie_interface_version
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_pcie_info_t_pcie_metric_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: pcie_width
      :type:  Any


   .. py:attribute:: pcie_speed
      :type:  Any


   .. py:attribute:: pcie_bandwidth
      :type:  Any


   .. py:attribute:: pcie_replay_count
      :type:  Any


   .. py:attribute:: pcie_l0_to_recovery_count
      :type:  Any


   .. py:attribute:: pcie_replay_roll_over_count
      :type:  Any


   .. py:attribute:: pcie_nak_sent_count
      :type:  Any


   .. py:attribute:: pcie_nak_received_count
      :type:  Any


   .. py:attribute:: pcie_lc_perf_other_end_recovery_count
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_pcie_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   pcie information
       


   .. py:attribute:: pcie_static
      :type:  Any


   .. py:attribute:: pcie_metric
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_power_cap_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Power Cap Information
       


   .. py:attribute:: power_cap
      :type:  Any


   .. py:attribute:: default_power_cap
      :type:  Any


   .. py:attribute:: dpm_cap
      :type:  Any


   .. py:attribute:: min_power_cap
      :type:  Any


   .. py:attribute:: max_power_cap
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_power_cap_type_t

   Bases: :py:obj:`enum.IntEnum`


   Power Cap Package Power Tracking (PPT) type
       


   .. py:attribute:: AMDSMI_POWER_CAP_TYPE_PPT0
      :type:  int


   .. py:attribute:: AMDSMI_POWER_CAP_TYPE_PPT1
      :type:  int


.. py:class:: amdsmi_vbios_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   VBios Information
       


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: build_date
      :type:  Any


   .. py:attribute:: part_number
      :type:  Any


   .. py:attribute:: version
      :type:  Any


   .. py:attribute:: boot_firmware
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_cache_property_type_t

   Bases: :py:obj:`enum.IntEnum`


   cache properties
       


   .. py:attribute:: AMDSMI_CACHE_PROPERTY_ENABLED
      :type:  int


   .. py:attribute:: AMDSMI_CACHE_PROPERTY_DATA_CACHE
      :type:  int


   .. py:attribute:: AMDSMI_CACHE_PROPERTY_INST_CACHE
      :type:  int


   .. py:attribute:: AMDSMI_CACHE_PROPERTY_CPU_CACHE
      :type:  int


   .. py:attribute:: AMDSMI_CACHE_PROPERTY_SIMD_CACHE
      :type:  int


.. py:class:: amdsmi_gpu_cache_info_t_cache_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: cache_properties
      :type:  Any


   .. py:attribute:: cache_size
      :type:  Any


   .. py:attribute:: cache_level
      :type:  Any


   .. py:attribute:: max_num_cu_shared
      :type:  Any


   .. py:attribute:: num_cache_instance
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_gpu_cache_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   GPU Cache Information
       


   .. py:attribute:: num_cache_types
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_fw_info_t_fw_info_list_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fw_id
      :type:  Any


   .. py:attribute:: fw_version
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_fw_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Firmware Information
       


   .. py:attribute:: num_fw_info
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_asic_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   ASIC Information
       


   .. py:attribute:: market_name
      :type:  Any


   .. py:attribute:: vendor_id
      :type:  Any


   .. py:attribute:: vendor_name
      :type:  Any


   .. py:attribute:: subvendor_id
      :type:  Any


   .. py:attribute:: device_id
      :type:  Any


   .. py:attribute:: rev_id
      :type:  Any


   .. py:attribute:: asic_serial
      :type:  Any


   .. py:attribute:: oam_id
      :type:  Any


   .. py:attribute:: num_of_compute_units
      :type:  Any


   .. py:attribute:: target_graphics_version
      :type:  Any


   .. py:attribute:: subsystem_id
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_kfd_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Structure holds kfd information
       


   .. py:attribute:: kfd_id
      :type:  Any


   .. py:attribute:: node_id
      :type:  Any


   .. py:attribute:: current_partition_id
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nps_caps_t_nps_flags_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: nps1_cap
      :type:  Any


   .. py:attribute:: nps2_cap
      :type:  Any


   .. py:attribute:: nps4_cap
      :type:  Any


   .. py:attribute:: nps8_cap
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nps_caps_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This union holds memory partition bitmask.
       


   .. py:attribute:: nps_flags
      :type:  Any


   .. py:attribute:: nps_cap_mask
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_memory_partition_config_t_numa_range_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: memory_type
      :type:  Any


   .. py:attribute:: start
      :type:  Any


   .. py:attribute:: end
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_memory_partition_config_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Memory Partition Configuration.
   This structure is used to identify various memory partition configurations.


   .. py:attribute:: partition_caps
      :type:  Any


   .. py:attribute:: mp_mode
      :type:  Any


   .. py:attribute:: num_numa_ranges
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_accelerator_partition_profile_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Accelerator Partition Resource Profile
       


   .. py:attribute:: profile_type
      :type:  Any


   .. py:attribute:: num_partitions
      :type:  Any


   .. py:attribute:: memory_caps
      :type:  Any


   .. py:attribute:: profile_index
      :type:  Any


   .. py:attribute:: num_resources
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_accelerator_partition_resource_profile_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Accelerator Partition Resources.
   This struct is used to identify various partition resource profiles.


   .. py:attribute:: profile_index
      :type:  Any


   .. py:attribute:: resource_type
      :type:  Any


   .. py:attribute:: partition_resource
      :type:  Any


   .. py:attribute:: num_partitions_share_resource
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_accelerator_partition_profile_config_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Accelerator Partition Profile Configurations
       


   .. py:attribute:: num_profiles
      :type:  Any


   .. py:attribute:: num_resource_profiles
      :type:  Any


   .. py:attribute:: default_profile_index
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_link_type_t

   Bases: :py:obj:`enum.IntEnum`


   Link type
       


   .. py:attribute:: AMDSMI_LINK_TYPE_INTERNAL
      :type:  int


   .. py:attribute:: AMDSMI_LINK_TYPE_PCIE
      :type:  int


   .. py:attribute:: AMDSMI_LINK_TYPE_XGMI
      :type:  int


   .. py:attribute:: AMDSMI_LINK_TYPE_NOT_APPLICABLE
      :type:  int


   .. py:attribute:: AMDSMI_LINK_TYPE_UNKNOWN
      :type:  int


   .. py:attribute:: AMDSMI_LINK_TYPE_NUMA
      :type:  int


   .. py:attribute:: AMDSMI_LINK_TYPE_XNUMA
      :type:  int


.. py:class:: amdsmi_cpu_util_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds CPU utilization information.
       


   .. py:attribute:: cpu_util_total
      :type:  Any


   .. py:attribute:: cpu_util_user
      :type:  Any


   .. py:attribute:: cpu_util_nice
      :type:  Any


   .. py:attribute:: cpu_util_sys
      :type:  Any


   .. py:attribute:: cpu_util_irq
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_link_status_t

   Bases: :py:obj:`enum.IntEnum`


   Link Status
       


   .. py:attribute:: AMDSMI_LINK_STATUS_ENABLED
      :type:  int


   .. py:attribute:: AMDSMI_LINK_STATUS_DISABLED
      :type:  int


   .. py:attribute:: AMDSMI_LINK_STATUS_INACTIVE
      :type:  int


   .. py:attribute:: AMDSMI_LINK_STATUS_ERROR
      :type:  int


.. py:class:: amdsmi_link_metrics_t__links(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: bdf
      :type:  Any


   .. py:attribute:: bit_rate
      :type:  Any


   .. py:attribute:: max_bandwidth
      :type:  Any


   .. py:attribute:: link_type
      :type:  Any


   .. py:attribute:: read
      :type:  Any


   .. py:attribute:: write
      :type:  Any


   .. py:attribute:: link_status
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_link_metrics_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Link Metrics
       


   .. py:attribute:: num_links
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_vram_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   VRam Information
       


   .. py:attribute:: vram_type
      :type:  Any


   .. py:attribute:: vram_vendor
      :type:  Any


   .. py:attribute:: vram_size
      :type:  Any


   .. py:attribute:: vram_bit_width
      :type:  Any


   .. py:attribute:: vram_max_bandwidth
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_driver_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Driver Information
       


   .. py:attribute:: driver_version
      :type:  Any


   .. py:attribute:: driver_date
      :type:  Any


   .. py:attribute:: driver_name
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_board_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Board Information
       


   .. py:attribute:: model_number
      :type:  Any


   .. py:attribute:: product_serial
      :type:  Any


   .. py:attribute:: fru_id
      :type:  Any


   .. py:attribute:: product_name
      :type:  Any


   .. py:attribute:: manufacturer_name
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_power_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Power Information

   Note:
       Unsupported struct members are set to UINT32_MAX


   .. py:attribute:: socket_power
      :type:  Any


   .. py:attribute:: current_socket_power
      :type:  Any


   .. py:attribute:: average_socket_power
      :type:  Any


   .. py:attribute:: gfx_voltage
      :type:  Any


   .. py:attribute:: soc_voltage
      :type:  Any


   .. py:attribute:: mem_voltage
      :type:  Any


   .. py:attribute:: power_limit
      :type:  Any


   .. py:attribute:: ubb_power
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_clk_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Clock Information
       


   .. py:attribute:: clk
      :type:  Any


   .. py:attribute:: min_clk
      :type:  Any


   .. py:attribute:: max_clk
      :type:  Any


   .. py:attribute:: clk_locked
      :type:  Any


   .. py:attribute:: clk_deep_sleep
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_engine_usage_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Engine Usage
   amdsmi_engine_usage_t:
   This structure holds common
   GPU activity values seen in both BM or
   SRIOV


   .. py:attribute:: gfx_activity
      :type:  Any


   .. py:attribute:: umc_activity
      :type:  Any


   .. py:attribute:: mm_activity
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_proc_info_t_engine_usage_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: gfx
      :type:  Any


   .. py:attribute:: enc
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_proc_info_t_memory_usage_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: gtt_mem
      :type:  Any


   .. py:attribute:: cpu_mem
      :type:  Any


   .. py:attribute:: vram_mem
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_proc_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Process Information
       


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: pid
      :type:  Any


   .. py:attribute:: mem
      :type:  Any


   .. py:attribute:: engine_usage
      :type:  Any


   .. py:attribute:: memory_usage
      :type:  Any


   .. py:attribute:: container_name
      :type:  Any


   .. py:attribute:: cu_occupancy
      :type:  Any


   .. py:attribute:: evicted_time
      :type:  Any


   .. py:attribute:: sdma_usage
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_proc_gpu_entry_t_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: gfx
      :type:  Any


   .. py:attribute:: enc
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_proc_gpu_entry_t_struct_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: gtt_mem
      :type:  Any


   .. py:attribute:: cpu_mem
      :type:  Any


   .. py:attribute:: vram_mem
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_proc_gpu_entry_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Per-GPU process entry within a PID-grouped result.
       


   .. py:attribute:: gpu_index
      :type:  Any


   .. py:attribute:: mem
      :type:  Any


   .. py:attribute:: engine_usage
      :type:  Any


   .. py:attribute:: memory_usage
      :type:  Any


   .. py:attribute:: cu_occupancy
      :type:  Any


   .. py:attribute:: evicted_time
      :type:  Any


   .. py:attribute:: sdma_usage
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_proc_info_by_pid_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Process info aggregated across all GPUs, keyed by PID.
       


   .. py:attribute:: pid
      :type:  Any


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: container_name
      :type:  Any


   .. py:attribute:: num_gpus
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_p2p_capability_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   IO Link P2P Capability
       


   .. py:attribute:: is_iolink_coherent
      :type:  Any


   .. py:attribute:: is_iolink_atomics_32bit
      :type:  Any


   .. py:attribute:: is_iolink_atomics_64bit
      :type:  Any


   .. py:attribute:: is_iolink_dma
      :type:  Any


   .. py:attribute:: is_iolink_bi_directional
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_dev_perf_level_t

   Bases: :py:obj:`enum.IntEnum`


   PowerPlay performance levels
       


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_AUTO
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_LOW
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_HIGH
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_MANUAL
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_STABLE_STD
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_DETERMINISM
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_LAST
      :type:  int


   .. py:attribute:: AMDSMI_DEV_PERF_LEVEL_UNKNOWN
      :type:  int


.. py:class:: amdsmi_event_group_t

   Bases: :py:obj:`enum.IntEnum`


   Event Groups
   Enum denoting an event group. The value of the enum is the
   base value for all the event enums in the group.


   .. py:attribute:: AMDSMI_EVNT_GRP_XGMI
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_GRP_XGMI_DATA_OUT
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_GRP_INVALID
      :type:  int


.. py:class:: amdsmi_event_type_t

   Bases: :py:obj:`enum.IntEnum`


   Event types
   Event type enum. Events belonging to a particular event group
   ::amdsmi_event_group_t should begin enumerating at the ::amdsmi_event_group_t
   value for that group.

   Data beats sent to neighbor 0; Each beat represents 32 bytes.

   XGMI throughput can be calculated by multiplying a BEATs event
   such as ::AMDSMI_EVNT_XGMI_0_BEATS_TX by 32 and dividing by
   the time for which event collection occurred,
   ::amdsmi_counter_value_t.time_running (which is in nanoseconds). To get
   bytes per second, multiply this value by 10<sup>9</sup>.

   Throughput = BEATS/time_running * 10<sup>9</sup>  (bytes/second)

   Events in the AMDSMI_EVNT_GRP_XGMI_DATA_OUT group measure
   the number of beats sent on an XGMI link. Each beat represents
   32 bytes. AMDSMI_EVNT_XGMI_DATA_OUT_n represents the number of
   outbound beats (each representing 32 bytes) on link n.

   XGMI throughput can be calculated by multiplying a event
   such as ::AMDSMI_EVNT_XGMI_DATA_OUT_n by 32 and dividing by
   the time for which event collection occurred,
   ::amdsmi_counter_value_t.time_running (which is in nanoseconds). To get
   bytes per second, multiply this value by 10<sup>9</sup>.


   .. py:attribute:: AMDSMI_EVNT_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_0_NOP_TX
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_0_REQUEST_TX
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_0_RESPONSE_TX
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_0_BEATS_TX
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_1_NOP_TX
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_1_REQUEST_TX
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_1_RESPONSE_TX
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_1_BEATS_TX
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_LAST
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_DATA_OUT_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_DATA_OUT_0
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_DATA_OUT_1
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_DATA_OUT_2
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_DATA_OUT_3
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_DATA_OUT_4
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_DATA_OUT_5
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_XGMI_DATA_OUT_LAST
      :type:  int


   .. py:attribute:: AMDSMI_EVNT_LAST
      :type:  int


.. py:class:: amdsmi_counter_command_t

   Bases: :py:obj:`enum.IntEnum`


   Event counter commands
       


   .. py:attribute:: AMDSMI_CNTR_CMD_START
      :type:  int


   .. py:attribute:: AMDSMI_CNTR_CMD_STOP
      :type:  int


.. py:class:: amdsmi_counter_value_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Counter value
       


   .. py:attribute:: value
      :type:  Any


   .. py:attribute:: time_enabled
      :type:  Any


   .. py:attribute:: time_running
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_evt_notification_type_t

   Bases: :py:obj:`enum.IntEnum`


   Event notification event types
       


   .. py:attribute:: AMDSMI_EVT_NOTIF_NONE
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_VMFAULT
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_THERMAL_THROTTLE
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_GPU_PRE_RESET
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_GPU_POST_RESET
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_MIGRATE_START
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_MIGRATE_END
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_PAGE_FAULT_START
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_PAGE_FAULT_END
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_QUEUE_EVICTION
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_QUEUE_RESTORE
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_PROCESS_START
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_PROCESS_END
      :type:  int


   .. py:attribute:: AMDSMI_EVT_NOTIF_LAST
      :type:  int


.. py:class:: amdsmi_evt_notification_data_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Event notification data returned from event notification API
       


   .. py:attribute:: processor_handle
      :type:  Any


   .. py:attribute:: event
      :type:  Any


   .. py:attribute:: message
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_temperature_metric_t

   Bases: :py:obj:`enum.IntEnum`


   Temperature Metrics. This enum is used to identify various
   temperature metrics. Corresponding values will be in Celsius


   .. py:attribute:: AMDSMI_TEMP_CURRENT
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_MAX
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_MIN
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_MAX_HYST
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_MIN_HYST
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_CRITICAL
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_CRITICAL_HYST
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_EMERGENCY
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_EMERGENCY_HYST
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_CRIT_MIN
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_CRIT_MIN_HYST
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_OFFSET
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_LOWEST
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_HIGHEST
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_SHUTDOWN
      :type:  int


   .. py:attribute:: AMDSMI_TEMP_LAST
      :type:  int


.. py:class:: amdsmi_voltage_metric_t

   Bases: :py:obj:`enum.IntEnum`


   Voltage Metrics.  This enum is used to identify various
   Voltage metrics. Corresponding values will be in millivolt.


   .. py:attribute:: AMDSMI_VOLT_CURRENT
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_MAX
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_MIN_CRIT
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_MIN
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_MAX_CRIT
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_AVERAGE
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_LOWEST
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_HIGHEST
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_LAST
      :type:  int


.. py:class:: amdsmi_voltage_type_t

   Bases: :py:obj:`enum.IntEnum`


   This ennumeration is used to indicate which type of
   voltage reading should be obtained.


   .. py:attribute:: AMDSMI_VOLT_TYPE_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_TYPE_VDDGFX
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_TYPE_VDDBOARD
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_TYPE_LAST
      :type:  int


   .. py:attribute:: AMDSMI_VOLT_TYPE_INVALID
      :type:  int


.. py:class:: amdsmi_power_profile_preset_masks_t

   Bases: :py:obj:`enum.IntEnum`


   Pre-set Profile Selections. These bitmasks can be AND'd with the
   ::amdsmi_power_profile_status_t.available_profiles returned from
   :: amdsmi_get_gpu_power_profile_presets to determine which power profiles
   are supported by the system.


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_CUSTOM_MASK
      :type:  int


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_VIDEO_MASK
      :type:  int


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK
      :type:  int


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_COMPUTE_MASK
      :type:  int


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_VR_MASK
      :type:  int


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK
      :type:  int


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT
      :type:  int


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_LAST
      :type:  int


   .. py:attribute:: AMDSMI_PWR_PROF_PRST_INVALID
      :type:  int


.. py:class:: amdsmi_gpu_block_t

   Bases: :py:obj:`enum.IntEnum`


   This enum is used to identify different GPU blocks.
       


   .. py:attribute:: AMDSMI_GPU_BLOCK_INVALID
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_UMC
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_SDMA
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_GFX
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_MMHUB
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_ATHUB
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_PCIE_BIF
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_HDP
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_XGMI_WAFL
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_DF
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_SMN
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_SEM
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_MP0
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_MP1
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_FUSE
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_MCA
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_VCN
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_JPEG
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_IH
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_MPIO
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_LAST
      :type:  int


   .. py:attribute:: AMDSMI_GPU_BLOCK_RESERVED
      :type:  int


.. py:class:: amdsmi_clk_limit_type_t

   Bases: :py:obj:`enum.IntEnum`


   The clk limit type
       


   .. py:attribute:: AMDSMI_CLK_LIMIT_MIN
      :type:  int


   .. py:attribute:: AMDSMI_CLK_LIMIT_MAX
      :type:  int


   .. py:attribute:: CLK_LIMIT_MIN
      :type:  int


   .. py:attribute:: CLK_LIMIT_MAX
      :type:  int


.. py:class:: amdsmi_cper_sev_t

   Bases: :py:obj:`enum.IntEnum`


   Cper sev
       


   .. py:attribute:: AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED
      :type:  int


   .. py:attribute:: AMDSMI_CPER_SEV_FATAL
      :type:  int


   .. py:attribute:: AMDSMI_CPER_SEV_NON_FATAL_CORRECTED
      :type:  int


   .. py:attribute:: AMDSMI_CPER_SEV_NUM
      :type:  int


   .. py:attribute:: AMDSMI_CPER_SEV_UNUSED
      :type:  int


.. py:class:: amdsmi_cper_notify_type_t

   Bases: :py:obj:`enum.IntEnum`


   Cper notify
       


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_CMC
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_CPE
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_MCE
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_PCIE
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_INIT
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_NMI
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_BOOT
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_DMAR
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_SEA
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_SEI
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_PEI
      :type:  int


   .. py:attribute:: AMDSMI_CPER_NOTIFY_TYPE_CXL_COMPONENT
      :type:  int


.. py:class:: amdsmi_gpu_ras_policy_v4_0_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Ras policy v4.0
       


   .. py:attribute:: dram_non_critical_region_threshold
      :type:  Any


   .. py:attribute:: dram_critical_region_threshold
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_gpu_ras_policy_info_t_policy_data_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: v4_0
      :type:  Any


   .. py:attribute:: info
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_gpu_ras_policy_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Ras policy info structure for storing version and different ras
   policy version structures


   .. py:attribute:: major_version
      :type:  Any


   .. py:attribute:: minor_version
      :type:  Any


   .. py:attribute:: policy_data
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_ras_err_state_t

   Bases: :py:obj:`enum.IntEnum`


   The current ECC state
       


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_NONE
      :type:  int


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_DISABLED
      :type:  int


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_PARITY
      :type:  int


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_SING_C
      :type:  int


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_MULT_UC
      :type:  int


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_POISON
      :type:  int


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_ENABLED
      :type:  int


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_LAST
      :type:  int


   .. py:attribute:: AMDSMI_RAS_ERR_STATE_INVALID
      :type:  int


.. py:class:: amdsmi_memory_type_t

   Bases: :py:obj:`enum.IntEnum`


   Types of memory

   Note:
       Sum of the process memory is not expected to be the total memory usage.


   .. py:attribute:: AMDSMI_MEM_TYPE_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_MEM_TYPE_VRAM
      :type:  int


   .. py:attribute:: AMDSMI_MEM_TYPE_VIS_VRAM
      :type:  int


   .. py:attribute:: AMDSMI_MEM_TYPE_GTT
      :type:  int


   .. py:attribute:: AMDSMI_MEM_TYPE_LAST
      :type:  int


.. py:class:: amdsmi_freq_ind_t

   Bases: :py:obj:`enum.IntEnum`


   The values of this enum are used as frequency identifiers.
       


   .. py:attribute:: AMDSMI_FREQ_IND_MIN
      :type:  int


   .. py:attribute:: AMDSMI_FREQ_IND_MAX
      :type:  int


   .. py:attribute:: AMDSMI_FREQ_IND_INVALID
      :type:  int


.. py:class:: amdsmi_xgmi_status_t

   Bases: :py:obj:`enum.IntEnum`


   XGMI Status
       


   .. py:attribute:: AMDSMI_XGMI_STATUS_NO_ERRORS
      :type:  int


   .. py:attribute:: AMDSMI_XGMI_STATUS_ERROR
      :type:  int


   .. py:attribute:: AMDSMI_XGMI_STATUS_MULTIPLE_ERRORS
      :type:  int


.. py:class:: amdsmi_memory_page_status_t

   Bases: :py:obj:`enum.IntEnum`


   Reserved Memory Page States
       


   .. py:attribute:: AMDSMI_MEM_PAGE_STATUS_RESERVED
      :type:  int


   .. py:attribute:: AMDSMI_MEM_PAGE_STATUS_PENDING
      :type:  int


   .. py:attribute:: AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE
      :type:  int


.. py:class:: amdsmi_utilization_counter_type_t

   Bases: :py:obj:`enum.IntEnum`


   The utilization counter type
       


   .. py:attribute:: AMDSMI_UTILIZATION_COUNTER_FIRST
      :type:  int


   .. py:attribute:: AMDSMI_COARSE_GRAIN_GFX_ACTIVITY
      :type:  int


   .. py:attribute:: AMDSMI_COARSE_GRAIN_MEM_ACTIVITY
      :type:  int


   .. py:attribute:: AMDSMI_COARSE_DECODER_ACTIVITY
      :type:  int


   .. py:attribute:: AMDSMI_FINE_GRAIN_GFX_ACTIVITY
      :type:  int


   .. py:attribute:: AMDSMI_FINE_GRAIN_MEM_ACTIVITY
      :type:  int


   .. py:attribute:: AMDSMI_FINE_DECODER_ACTIVITY
      :type:  int


   .. py:attribute:: AMDSMI_UTILIZATION_COUNTER_LAST
      :type:  int


.. py:class:: amdsmi_utilization_counter_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   The utilization counter data

   The max number of values per counter type


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: value
      :type:  Any


   .. py:attribute:: fine_value
      :type:  Any


   .. py:attribute:: fine_value_count
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_retired_page_record_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Reserved Memory Page Record
       


   .. py:attribute:: page_address
      :type:  Any


   .. py:attribute:: page_size
      :type:  Any


   .. py:attribute:: status
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_power_profile_status_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure contains information about which power profiles are
   supported by the system for a given device, and which power profile is
   currently active.


   .. py:attribute:: available_profiles
      :type:  Any


   .. py:attribute:: current
      :type:  Any


   .. py:attribute:: num_profiles
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_frequencies_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds information about clock frequencies.
       


   .. py:attribute:: has_deep_sleep
      :type:  Any


   .. py:attribute:: num_supported
      :type:  Any


   .. py:attribute:: current
      :type:  Any


   .. py:attribute:: frequency
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_dpm_policy_entry_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   The dpm policy.
       


   .. py:attribute:: policy_id
      :type:  Any


   .. py:attribute:: policy_description
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_dpm_policy_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   DPM Policy

   Only the first num_supported policies are valid.


   .. py:attribute:: num_supported
      :type:  Any


   .. py:attribute:: current
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_pcie_bandwidth_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds information about the possible PCIe
   bandwidths. Specifically, the possible transfer rates and their
   associated numbers of lanes are stored here.

   Only the first num_supported bandwidths are valid.


   .. py:attribute:: transfer_rate
      :type:  Any


   .. py:attribute:: lanes
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_version_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds version information.
       


   .. py:attribute:: major
      :type:  Any


   .. py:attribute:: minor
      :type:  Any


   .. py:attribute:: release
      :type:  Any


   .. py:attribute:: build
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_od_vddc_point_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure represents a point on the frequency-voltage plane.
       


   .. py:attribute:: frequency
      :type:  Any


   .. py:attribute:: voltage
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_freq_volt_region_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds 2 ::amdsmi_range_t's, one for frequency and one for
   voltage. These 2 ranges indicate the range of possible values for the
   corresponding ::amdsmi_od_vddc_point_t.


   .. py:attribute:: freq_range
      :type:  Any


   .. py:attribute:: volt_range
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_od_volt_curve_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   OD Vold Curve
   ::AMDSMI_NUM_VOLTAGE_CURVE_POINTS number of ::amdsmi_od_vddc_point_t's


.. py:class:: amdsmi_od_volt_freq_data_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds the frequency-voltage values for a device.
       


   .. py:attribute:: curr_sclk_range
      :type:  Any


   .. py:attribute:: curr_mclk_range
      :type:  Any


   .. py:attribute:: curr_fclk_range
      :type:  Any


   .. py:attribute:: sclk_freq_limits
      :type:  Any


   .. py:attribute:: mclk_freq_limits
      :type:  Any


   .. py:attribute:: fclk_freq_limits
      :type:  Any


   .. py:attribute:: curve
      :type:  Any


   .. py:attribute:: num_regions
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amd_metrics_table_header_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Structure holds the gpu metrics table header for a device

   Size and version information of metrics data


   .. py:attribute:: structure_size
      :type:  Any


   .. py:attribute:: format_revision
      :type:  Any


   .. py:attribute:: content_revision
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_gpu_xcp_metrics_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   The following structures hold the gpu statistics for a device.
       


   .. py:attribute:: gfx_busy_inst
      :type:  Any


   .. py:attribute:: jpeg_busy
      :type:  Any


   .. py:attribute:: vcn_busy
      :type:  Any


   .. py:attribute:: gfx_busy_acc
      :type:  Any


   .. py:attribute:: gfx_below_host_limit_acc
      :type:  Any


   .. py:attribute:: gfx_below_host_limit_ppt_acc
      :type:  Any


   .. py:attribute:: gfx_below_host_limit_thm_acc
      :type:  Any


   .. py:attribute:: gfx_low_utilization_acc
      :type:  Any


   .. py:attribute:: gfx_below_host_limit_total_acc
      :type:  Any


   .. py:attribute:: temperature_xcd
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_apu_metrics_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   APU metrics auxiliary data.

   This structure holds unified APU-specific metrics data derived from the
   underlying driver metrics table. It is attached via
   ::amdsmi_gpu_metrics_t.apu_metrics when APU-specific metrics are available.

   **Version Support:**
   - v2.4: format_revision == 2 && content_revision == 4
   - v3.0: format_revision == 3 && content_revision == 0
   Use ::amdsmi_gpu_metrics_t.common_header to identify which version populated
   the fields.

   **Sentinel Values:**
   Fields not applicable to the current version are initialized to the maximum value
   of their respective type: 0xFFFF for uint16_t fields, 0xFFFFFFFF for uint32_t fields,
   and UINT64_MAX for uint64_t fields. For example, on v3.0 hardware, v2.4-only fields
   like `average_mm_activity` and `temperature_l3` will contain 0xFFFF. Similarly,
   array elements beyond the version-specific count (e.g., elements 8-15 of
   `temperature_core` on v2.4) will contain 0xFFFF. However, uint32_t elements such as
   'throttle_status' will contain 0xFFFFFFFF and UINT64_MAX for uint64_t elements such
   as 'indep_throttle_status'. Callers should check the version and treat maximum values
   as invalid/not applicable.


   .. py:attribute:: temperature_gfx
      :type:  Any


   .. py:attribute:: temperature_soc
      :type:  Any


   .. py:attribute:: temperature_core
      :type:  Any


   .. py:attribute:: temperature_l3
      :type:  Any


   .. py:attribute:: temperature_skin
      :type:  Any


   .. py:attribute:: average_gfx_activity
      :type:  Any


   .. py:attribute:: average_mm_activity
      :type:  Any


   .. py:attribute:: average_vcn_activity
      :type:  Any


   .. py:attribute:: average_ipu_activity
      :type:  Any


   .. py:attribute:: average_core_c0_activity
      :type:  Any


   .. py:attribute:: average_dram_reads
      :type:  Any


   .. py:attribute:: average_dram_writes
      :type:  Any


   .. py:attribute:: average_ipu_reads
      :type:  Any


   .. py:attribute:: average_ipu_writes
      :type:  Any


   .. py:attribute:: average_socket_power
      :type:  Any


   .. py:attribute:: average_cpu_power
      :type:  Any


   .. py:attribute:: average_soc_power
      :type:  Any


   .. py:attribute:: average_gfx_power
      :type:  Any


   .. py:attribute:: average_core_power
      :type:  Any


   .. py:attribute:: average_ipu_power
      :type:  Any


   .. py:attribute:: average_apu_power
      :type:  Any


   .. py:attribute:: average_dgpu_power
      :type:  Any


   .. py:attribute:: average_all_core_power
      :type:  Any


   .. py:attribute:: average_sys_power
      :type:  Any


   .. py:attribute:: stapm_power_limit
      :type:  Any


   .. py:attribute:: current_stapm_power_limit
      :type:  Any


   .. py:attribute:: average_gfxclk_frequency
      :type:  Any


   .. py:attribute:: average_socclk_frequency
      :type:  Any


   .. py:attribute:: average_uclk_frequency
      :type:  Any


   .. py:attribute:: average_fclk_frequency
      :type:  Any


   .. py:attribute:: average_vclk_frequency
      :type:  Any


   .. py:attribute:: average_dclk_frequency
      :type:  Any


   .. py:attribute:: average_vpeclk_frequency
      :type:  Any


   .. py:attribute:: average_ipuclk_frequency
      :type:  Any


   .. py:attribute:: average_mpipu_frequency
      :type:  Any


   .. py:attribute:: current_gfxclk
      :type:  Any


   .. py:attribute:: current_socclk
      :type:  Any


   .. py:attribute:: current_uclk
      :type:  Any


   .. py:attribute:: current_fclk
      :type:  Any


   .. py:attribute:: current_vclk
      :type:  Any


   .. py:attribute:: current_dclk
      :type:  Any


   .. py:attribute:: current_coreclk
      :type:  Any


   .. py:attribute:: current_l3clk
      :type:  Any


   .. py:attribute:: current_core_maxfreq
      :type:  Any


   .. py:attribute:: current_gfx_maxfreq
      :type:  Any


   .. py:attribute:: throttle_status
      :type:  Any


   .. py:attribute:: indep_throttle_status
      :type:  Any


   .. py:attribute:: throttle_residency_prochot
      :type:  Any


   .. py:attribute:: throttle_residency_spl
      :type:  Any


   .. py:attribute:: throttle_residency_fppt
      :type:  Any


   .. py:attribute:: throttle_residency_sppt
      :type:  Any


   .. py:attribute:: throttle_residency_thm_core
      :type:  Any


   .. py:attribute:: throttle_residency_thm_gfx
      :type:  Any


   .. py:attribute:: throttle_residency_thm_soc
      :type:  Any


   .. py:attribute:: fan_pwm
      :type:  Any


   .. py:attribute:: average_temperature_gfx
      :type:  Any


   .. py:attribute:: average_temperature_soc
      :type:  Any


   .. py:attribute:: average_temperature_core
      :type:  Any


   .. py:attribute:: average_temperature_l3
      :type:  Any


   .. py:attribute:: average_cpu_voltage
      :type:  Any


   .. py:attribute:: average_soc_voltage
      :type:  Any


   .. py:attribute:: average_gfx_voltage
      :type:  Any


   .. py:attribute:: average_cpu_current
      :type:  Any


   .. py:attribute:: average_soc_current
      :type:  Any


   .. py:attribute:: average_gfx_current
      :type:  Any


   .. py:attribute:: time_filter_alphavalue
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_gpu_metrics_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Structure holds the gpu metrics values for a device

   This structure is extended to fit the needs of different GPU metric
   versions when exposing data through the structure.
   Depending on the version, some data members will hold data, and
   some will not. A good example is the set of 'current clocks':
   current_gfxclk, current_socclk, current_vclk0, current_dclk0.
   These are single-valued data members, up to version 1.3.
   For version 1.4 and up these are multi-valued data members (arrays)
   and their counterparts; current_gfxclks[], current_socclks[],
   current_vclk0s[], current_dclk0s[], will hold the data


   .. py:attribute:: common_header
      :type:  Any


   .. py:attribute:: temperature_edge
      :type:  Any


   .. py:attribute:: temperature_hotspot
      :type:  Any


   .. py:attribute:: temperature_mem
      :type:  Any


   .. py:attribute:: temperature_vrgfx
      :type:  Any


   .. py:attribute:: temperature_vrsoc
      :type:  Any


   .. py:attribute:: temperature_vrmem
      :type:  Any


   .. py:attribute:: average_gfx_activity
      :type:  Any


   .. py:attribute:: average_umc_activity
      :type:  Any


   .. py:attribute:: average_mm_activity
      :type:  Any


   .. py:attribute:: average_socket_power
      :type:  Any


   .. py:attribute:: energy_accumulator
      :type:  Any


   .. py:attribute:: system_clock_counter
      :type:  Any


   .. py:attribute:: average_gfxclk_frequency
      :type:  Any


   .. py:attribute:: average_socclk_frequency
      :type:  Any


   .. py:attribute:: average_uclk_frequency
      :type:  Any


   .. py:attribute:: average_vclk0_frequency
      :type:  Any


   .. py:attribute:: average_dclk0_frequency
      :type:  Any


   .. py:attribute:: average_vclk1_frequency
      :type:  Any


   .. py:attribute:: average_dclk1_frequency
      :type:  Any


   .. py:attribute:: current_gfxclk
      :type:  Any


   .. py:attribute:: current_socclk
      :type:  Any


   .. py:attribute:: current_uclk
      :type:  Any


   .. py:attribute:: current_vclk0
      :type:  Any


   .. py:attribute:: current_dclk0
      :type:  Any


   .. py:attribute:: current_vclk1
      :type:  Any


   .. py:attribute:: current_dclk1
      :type:  Any


   .. py:attribute:: throttle_status
      :type:  Any


   .. py:attribute:: current_fan_speed
      :type:  Any


   .. py:attribute:: pcie_link_width
      :type:  Any


   .. py:attribute:: pcie_link_speed
      :type:  Any


   .. py:attribute:: gfx_activity_acc
      :type:  Any


   .. py:attribute:: mem_activity_acc
      :type:  Any


   .. py:attribute:: temperature_hbm
      :type:  Any


   .. py:attribute:: firmware_timestamp
      :type:  Any


   .. py:attribute:: voltage_soc
      :type:  Any


   .. py:attribute:: voltage_gfx
      :type:  Any


   .. py:attribute:: voltage_mem
      :type:  Any


   .. py:attribute:: indep_throttle_status
      :type:  Any


   .. py:attribute:: current_socket_power
      :type:  Any


   .. py:attribute:: vcn_activity
      :type:  Any


   .. py:attribute:: gfxclk_lock_status
      :type:  Any


   .. py:attribute:: xgmi_link_width
      :type:  Any


   .. py:attribute:: xgmi_link_speed
      :type:  Any


   .. py:attribute:: pcie_bandwidth_acc
      :type:  Any


   .. py:attribute:: pcie_bandwidth_inst
      :type:  Any


   .. py:attribute:: pcie_l0_to_recov_count_acc
      :type:  Any


   .. py:attribute:: pcie_replay_count_acc
      :type:  Any


   .. py:attribute:: pcie_replay_rover_count_acc
      :type:  Any


   .. py:attribute:: xgmi_read_data_acc
      :type:  Any


   .. py:attribute:: xgmi_write_data_acc
      :type:  Any


   .. py:attribute:: current_gfxclks
      :type:  Any


   .. py:attribute:: current_socclks
      :type:  Any


   .. py:attribute:: current_vclk0s
      :type:  Any


   .. py:attribute:: current_dclk0s
      :type:  Any


   .. py:attribute:: jpeg_activity
      :type:  Any


   .. py:attribute:: pcie_nak_sent_count_acc
      :type:  Any


   .. py:attribute:: pcie_nak_rcvd_count_acc
      :type:  Any


   .. py:attribute:: accumulation_counter
      :type:  Any


   .. py:attribute:: prochot_residency_acc
      :type:  Any


   .. py:attribute:: ppt_residency_acc
      :type:  Any


   .. py:attribute:: socket_thm_residency_acc
      :type:  Any


   .. py:attribute:: vr_thm_residency_acc
      :type:  Any


   .. py:attribute:: hbm_thm_residency_acc
      :type:  Any


   .. py:attribute:: num_partition
      :type:  Any


   .. py:attribute:: pcie_lc_perf_other_end_recovery
      :type:  Any


   .. py:attribute:: vram_max_bandwidth
      :type:  Any


   .. py:attribute:: xgmi_link_status
      :type:  Any


   .. py:attribute:: temperature_hbm_stacks
      :type:  Any


   .. py:attribute:: temperature_mid
      :type:  Any


   .. py:attribute:: temperature_aid
      :type:  Any


   .. py:attribute:: current_uclk_aid
      :type:  Any


   .. py:attribute:: current_socclks_mid
      :type:  Any


   .. py:attribute:: apu_metrics
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_xgmi_link_status_type_t

   Bases: :py:obj:`enum.IntEnum`


   XGMI Link Status Type
       


   .. py:attribute:: AMDSMI_XGMI_LINK_DOWN
      :type:  int


   .. py:attribute:: AMDSMI_XGMI_LINK_UP
      :type:  int


   .. py:attribute:: AMDSMI_XGMI_LINK_DISABLE
      :type:  int


.. py:class:: amdsmi_xgmi_link_status_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   XGMI Link Status
       


   .. py:attribute:: total_links
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_name_value_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds the name value pairs
       


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: value
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_reg_type_t

   Bases: :py:obj:`enum.IntEnum`


   This register type for register table
       


   .. py:attribute:: AMDSMI_REG_XGMI
      :type:  int


   .. py:attribute:: AMDSMI_REG_WAFL
      :type:  int


   .. py:attribute:: AMDSMI_REG_PCIE
      :type:  int


   .. py:attribute:: AMDSMI_REG_USR
      :type:  int


   .. py:attribute:: AMDSMI_REG_USR1
      :type:  int


.. py:class:: amdsmi_ras_feature_t_ras_info_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: dram_ecc
      :type:  Any


   .. py:attribute:: sram_ecc
      :type:  Any


   .. py:attribute:: poisoning
      :type:  Any


   .. py:attribute:: rsvd
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_ras_feature_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds ras feature information.
       


   .. py:attribute:: ras_eeprom_version
      :type:  Any


   .. py:attribute:: ecc_correction_schema_flag
      :type:  Any


   .. py:attribute:: ras_info
      :type:  Any


   .. py:attribute:: needs_reboot
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_error_count_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds error counts.
       


   .. py:attribute:: correctable_count
      :type:  Any


   .. py:attribute:: uncorrectable_count
      :type:  Any


   .. py:attribute:: deferred_count
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_process_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure contains information specific to a process.
   Sum of the process memory is not expected to be the total memory usage.


   .. py:attribute:: process_id
      :type:  Any


   .. py:attribute:: vram_usage
      :type:  Any


   .. py:attribute:: sdma_usage
      :type:  Any


   .. py:attribute:: cu_occupancy
      :type:  Any


   .. py:attribute:: evicted_time
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_topology_nearest_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Topology Nearest
       


   .. py:attribute:: count
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_virtualization_mode_t

   Bases: :py:obj:`enum.IntEnum`


   Variant placeholder

   Place-holder "variant" for functions that have don't have any variants,
   but do have monitors or sensors.


   .. py:attribute:: AMDSMI_VIRTUALIZATION_MODE_UNKNOWN
      :type:  int


   .. py:attribute:: AMDSMI_VIRTUALIZATION_MODE_BAREMETAL
      :type:  int


   .. py:attribute:: AMDSMI_VIRTUALIZATION_MODE_HOST
      :type:  int


   .. py:attribute:: AMDSMI_VIRTUALIZATION_MODE_GUEST
      :type:  int


   .. py:attribute:: AMDSMI_VIRTUALIZATION_MODE_PASSTHROUGH
      :type:  int


.. py:class:: amdsmi_affinity_scope_t

   Bases: :py:obj:`enum.IntEnum`


   Scope for Numa affinity or Socket affinity
       


   .. py:attribute:: AMDSMI_AFFINITY_SCOPE_NODE
      :type:  int


   .. py:attribute:: AMDSMI_AFFINITY_SCOPE_SOCKET
      :type:  int


.. py:class:: amdsmi_npm_status_t

   Bases: :py:obj:`enum.IntEnum`


   NPM status
       


   .. py:attribute:: AMDSMI_NPM_STATUS_DISABLED
      :type:  int


   .. py:attribute:: AMDSMI_NPM_STATUS_ENABLED
      :type:  int


.. py:class:: amdsmi_npm_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NPM info
       


   .. py:attribute:: status
      :type:  Any


   .. py:attribute:: limit
      :type:  Any


   .. py:attribute:: ubb_power_threshold
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_ptl_data_format_t

   Bases: :py:obj:`enum.IntEnum`


   PTL (Peak Tops Limiter) data format types
   These correspond to the hardware data types used in matrix operations.
   Only F8 and XF32 are always supported at full performance. From the remaining
   five types, only two can be supported at peak performance simultaneously.


   .. py:attribute:: AMDSMI_PTL_DATA_FORMAT_I8
      :type:  int


   .. py:attribute:: AMDSMI_PTL_DATA_FORMAT_F16
      :type:  int


   .. py:attribute:: AMDSMI_PTL_DATA_FORMAT_BF16
      :type:  int


   .. py:attribute:: AMDSMI_PTL_DATA_FORMAT_F32
      :type:  int


   .. py:attribute:: AMDSMI_PTL_DATA_FORMAT_F64
      :type:  int


   .. py:attribute:: AMDSMI_PTL_DATA_FORMAT_F8
      :type:  int


   .. py:attribute:: AMDSMI_PTL_DATA_FORMAT_VECTOR
      :type:  int


   .. py:attribute:: AMDSMI_PTL_DATA_FORMAT_INVALID
      :type:  int


.. py:class:: amdsmi_smu_fw_version_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   This structure holds SMU Firmware version information.
       


   .. py:attribute:: debug
      :type:  Any


   .. py:attribute:: minor
      :type:  Any


   .. py:attribute:: major
      :type:  Any


   .. py:attribute:: unused
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_ddr_bw_metrics_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   DDR bandwidth metrics.
       


   .. py:attribute:: max_bw
      :type:  Any


   .. py:attribute:: utilized_bw
      :type:  Any


   .. py:attribute:: utilized_pct
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_temp_range_refresh_rate_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   temperature range and refresh rate metrics of a DIMM
       


   .. py:attribute:: range
      :type:  Any


   .. py:attribute:: ref_rate
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_dimm_power_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   DIMM Power(mW), power update rate(ms) and dimm address
       


   .. py:attribute:: power
      :type:  Any


   .. py:attribute:: update_rate
      :type:  Any


   .. py:attribute:: dimm_addr
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_dimm_thermal_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   DIMM temperature(°C) and update rate(ms) and dimm address
       


   .. py:attribute:: sensor
      :type:  Any


   .. py:attribute:: update_rate
      :type:  Any


   .. py:attribute:: dimm_addr
      :type:  Any


   .. py:attribute:: temp
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_io_bw_encoding_t

   Bases: :py:obj:`enum.IntEnum`


   xGMI Bandwidth Encoding types
       


   .. py:attribute:: AMDSMI_AGG_BW0
      :type:  int


   .. py:attribute:: AMDSMI_RD_BW0
      :type:  int


   .. py:attribute:: AMDSMI_WR_BW0
      :type:  int


   .. py:attribute:: AGG_BW0
      :type:  int


   .. py:attribute:: RD_BW0
      :type:  int


   .. py:attribute:: WR_BW0
      :type:  int


.. py:class:: amdsmi_link_id_bw_type_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   LINK name and Bandwidth type Information.It contains
   link names i.e valid link names are
   "P0", "P1", "P2", "P3", "P4", "G0", "G1", "G2", "G3", "G4"
   "G5", "G6", "G7"
   Valid bandwidth types 1(Aggregate_BW), 2 (Read BW), 4 (Write BW).


   .. py:attribute:: bw_type
      :type:  Any


   .. py:attribute:: link_name
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_dpm_level_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   max and min LCLK DPM level on a given NBIO ID.
   Valid max and min DPM level values are 0 - 1.


   .. py:attribute:: max_dpm_level
      :type:  Any


   .. py:attribute:: min_dpm_level
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_hsmp_metrics_table_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   HSMP Metrics table (supported only with hsmp proto version 6).
       


   .. py:attribute:: accumulation_counter
      :type:  Any


   .. py:attribute:: max_socket_temperature
      :type:  Any


   .. py:attribute:: max_vr_temperature
      :type:  Any


   .. py:attribute:: max_hbm_temperature
      :type:  Any


   .. py:attribute:: max_socket_temperature_acc
      :type:  Any


   .. py:attribute:: max_vr_temperature_acc
      :type:  Any


   .. py:attribute:: max_hbm_temperature_acc
      :type:  Any


   .. py:attribute:: socket_power_limit
      :type:  Any


   .. py:attribute:: max_socket_power_limit
      :type:  Any


   .. py:attribute:: socket_power
      :type:  Any


   .. py:attribute:: timestamp
      :type:  Any


   .. py:attribute:: socket_energy_acc
      :type:  Any


   .. py:attribute:: ccd_energy_acc
      :type:  Any


   .. py:attribute:: xcd_energy_acc
      :type:  Any


   .. py:attribute:: aid_energy_acc
      :type:  Any


   .. py:attribute:: hbm_energy_acc
      :type:  Any


   .. py:attribute:: cclk_frequency_limit
      :type:  Any


   .. py:attribute:: gfxclk_frequency_limit
      :type:  Any


   .. py:attribute:: fclk_frequency
      :type:  Any


   .. py:attribute:: uclk_frequency
      :type:  Any


   .. py:attribute:: socclk_frequency
      :type:  Any


   .. py:attribute:: vclk_frequency
      :type:  Any


   .. py:attribute:: dclk_frequency
      :type:  Any


   .. py:attribute:: lclk_frequency
      :type:  Any


   .. py:attribute:: gfxclk_frequency_acc
      :type:  Any


   .. py:attribute:: cclk_frequency_acc
      :type:  Any


   .. py:attribute:: max_cclk_frequency
      :type:  Any


   .. py:attribute:: min_cclk_frequency
      :type:  Any


   .. py:attribute:: max_gfxclk_frequency
      :type:  Any


   .. py:attribute:: min_gfxclk_frequency
      :type:  Any


   .. py:attribute:: fclk_frequency_table
      :type:  Any


   .. py:attribute:: uclk_frequency_table
      :type:  Any


   .. py:attribute:: socclk_frequency_table
      :type:  Any


   .. py:attribute:: vclk_frequency_table
      :type:  Any


   .. py:attribute:: dclk_frequency_table
      :type:  Any


   .. py:attribute:: lclk_frequency_table
      :type:  Any


   .. py:attribute:: max_lclk_dpm_range
      :type:  Any


   .. py:attribute:: min_lclk_dpm_range
      :type:  Any


   .. py:attribute:: xgmi_width
      :type:  Any


   .. py:attribute:: xgmi_bitrate
      :type:  Any


   .. py:attribute:: xgmi_read_bandwidth_acc
      :type:  Any


   .. py:attribute:: xgmi_write_bandwidth_acc
      :type:  Any


   .. py:attribute:: socket_c0_residency
      :type:  Any


   .. py:attribute:: socket_gfx_busy
      :type:  Any


   .. py:attribute:: dram_bandwidth_utilization
      :type:  Any


   .. py:attribute:: socket_c0_residency_acc
      :type:  Any


   .. py:attribute:: socket_gfx_busy_acc
      :type:  Any


   .. py:attribute:: dram_bandwidth_acc
      :type:  Any


   .. py:attribute:: max_dram_bandwidth
      :type:  Any


   .. py:attribute:: dram_bandwidth_utilization_acc
      :type:  Any


   .. py:attribute:: pcie_bandwidth_acc
      :type:  Any


   .. py:attribute:: prochot_residency_acc
      :type:  Any


   .. py:attribute:: ppt_residency_acc
      :type:  Any


   .. py:attribute:: socket_thm_residency_acc
      :type:  Any


   .. py:attribute:: vr_thm_residency_acc
      :type:  Any


   .. py:attribute:: hbm_thm_residency_acc
      :type:  Any


   .. py:attribute:: spare
      :type:  Any


   .. py:attribute:: gfxclk_frequency
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_cpu_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   cpu info data
       


   .. py:attribute:: model_name
      :type:  Any


   .. py:attribute:: cpu_family_id
      :type:  Any


   .. py:attribute:: model_id
      :type:  Any


   .. py:attribute:: threads_per_core
      :type:  Any


   .. py:attribute:: cores_per_socket
      :type:  Any


   .. py:attribute:: frequency_boost
      :type:  Any


   .. py:attribute:: vendor_id
      :type:  Any


   .. py:attribute:: vendor_name
      :type:  Any


   .. py:attribute:: subvendor_id
      :type:  Any


   .. py:attribute:: device_id
      :type:  Any


   .. py:attribute:: rev_id
      :type:  Any


   .. py:attribute:: asic_serial
      :type:  Any


   .. py:attribute:: socket_id
      :type:  Any


   .. py:attribute:: core_id
      :type:  Any


   .. py:attribute:: num_of_cpu_cores
      :type:  Any


   .. py:attribute:: socket_count
      :type:  Any


   .. py:attribute:: core_count
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_sock_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   cpu socket info data
       


   .. py:attribute:: socket_id
      :type:  Any


   .. py:attribute:: cores_per_socket
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_stat_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Structure for NIC statistic name-value pairs

   This structure represents a single NIC statistic with its name and value.


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: value
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_asic_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC asic information
       


   .. py:attribute:: vendor_id
      :type:  Any


   .. py:attribute:: subvendor_id
      :type:  Any


   .. py:attribute:: device_id
      :type:  Any


   .. py:attribute:: subsystem_id
      :type:  Any


   .. py:attribute:: revision
      :type:  Any


   .. py:attribute:: permanent_address
      :type:  Any


   .. py:attribute:: product_name
      :type:  Any


   .. py:attribute:: part_number
      :type:  Any


   .. py:attribute:: serial_number
      :type:  Any


   .. py:attribute:: vendor_name
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_bus_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC bus information
       


   .. py:attribute:: bdf
      :type:  Any


   .. py:attribute:: max_pcie_width
      :type:  Any


   .. py:attribute:: max_pcie_speed
      :type:  Any


   .. py:attribute:: pcie_interface_version
      :type:  Any


   .. py:attribute:: slot_type
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_numa_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC NUMA information
       


   .. py:attribute:: node
      :type:  Any


   .. py:attribute:: affinity
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_fw_entry_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC firmware information
       


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: version
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_fw_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC firmware information collection
       


   .. py:attribute:: num_fw
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_port_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC port information

   Active FEC Modes:
   The active_fec field provides a bitmask representation of Active FEC (Active Forward Error
   Correction) modes. The bitmask values are derived from the `ethtool_fecparam` structure,
   specifically the `active_fec` field. Below are examples of the defined FEC modes:

   Examples:
   - ETHTOOL_FEC_NONE  (0x01)
   - ETHTOOL_FEC_AUTO  (0x02)
   - ETHTOOL_FEC_RS    (0x04)
   - ETHTOOL_FEC_BASER (0x08)
   - ETHTOOL_FEC_LLRS  (0x10)
   - ETHTOOL_FEC_OFF   (0x20)

   Note: These definitions are based on the latest available ethtool information. Users should
   verify if there are any updates or changes to these definitions in the relevant ethtool
   structure or field before implementing them in their code.


   .. py:attribute:: bdf
      :type:  Any


   .. py:attribute:: port_num
      :type:  Any


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: flavour
      :type:  Any


   .. py:attribute:: netdev
      :type:  Any


   .. py:attribute:: ifindex
      :type:  Any


   .. py:attribute:: mac_address
      :type:  Any


   .. py:attribute:: carrier
      :type:  Any


   .. py:attribute:: mtu
      :type:  Any


   .. py:attribute:: link_state
      :type:  Any


   .. py:attribute:: link_speed
      :type:  Any


   .. py:attribute:: active_fec
      :type:  Any


   .. py:attribute:: autoneg
      :type:  Any


   .. py:attribute:: pause_autoneg
      :type:  Any


   .. py:attribute:: pause_rx
      :type:  Any


   .. py:attribute:: pause_tx
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_port_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC port information collection
       


   .. py:attribute:: num_ports
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_driver_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC driver information
       


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: version
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_rdma_port_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC RDMA port information
       


   .. py:attribute:: netdev
      :type:  Any


   .. py:attribute:: state
      :type:  Any


   .. py:attribute:: rdma_port
      :type:  Any


   .. py:attribute:: max_mtu
      :type:  Any


   .. py:attribute:: active_mtu
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_rdma_dev_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC RDMA device information
       


   .. py:attribute:: rdma_dev
      :type:  Any


   .. py:attribute:: node_guid
      :type:  Any


   .. py:attribute:: node_type
      :type:  Any


   .. py:attribute:: sys_image_guid
      :type:  Any


   .. py:attribute:: fw_ver
      :type:  Any


   .. py:attribute:: num_rdma_ports
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_nic_rdma_devices_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   NIC RDMA devices information collection
       


   .. py:attribute:: num_rdma_dev
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: amdsmi_init(init_flags)

   Initialize the AMD SMI library

   @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf}
   @platform{guest_mvf} @platform{guest_windows}

   This function initializes the library and the internal data structures,
   including those corresponding to sources of information that SMI provides.
   Singleton Design, requires the same number of inits as shutdowns.

   The ``init_flags`` decides which type of processor
   can be discovered by ::amdsmi_get_socket_handles(). AMDSMI_INIT_AMD_GPUS returns
   sockets with AMD GPUS, and AMDSMI_INIT_AMD_GPUS | AMDSMI_INIT_AMD_CPUS returns
   sockets with either AMD GPUS or CPUS.
   Both AMDSMI_INIT_AMD_GPUS and AMDSMI_INIT_AMD_CPUS flags are supported.

   Args:
       init_flags (:py:obj:`~.int`) -- *IN*:
           Bit flags that tell SMI how to initialize. Values of
           ::amdsmi_init_flags_t may be OR'd together and passed through ``init_flags``
           to modify how AMDSMI initializes.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_init(uint64_t init_flags)


.. py:function:: amdsmi_shut_down()

   Shutdown the AMD SMI library

   @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf}
   @platform{guest_mvf} @platform{guest_windows}

   This function shuts down the library and internal data structures and
   performs any necessary clean ups. Singleton Design, requires the same number
   of inits as shutdowns.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_shut_down()


.. py:function:: amdsmi_get_socket_handles(socket_count, socket_handles)

   Get the list of socket handles in the system.

   @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf}
   @platform{guest_mvf} @platform{guest_windows}

   Depends on what flag is passed to ::amdsmi_init.  AMDSMI_INIT_AMD_GPUS
   returns sockets with AMD GPUS, and AMDSMI_INIT_AMD_GPUS | AMDSMI_INIT_AMD_CPUS returns
   sockets with either AMD GPUS or CPUS.
   The socket handles can be used to query the processor handles in that socket, which
   will be used in other APIs to get processor detail information or telemtries.

   Args:
       socket_count (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           As input, the value passed
           through this parameter is the number of ::amdsmi_socket_handle that
           may be safely written to the memory pointed to by ``socket_handles.`` This is the
           limit on how many socket handles will be written to ``socket_handles.`` On return, `socket_count` will contain the number of socket handles written to ``socket_handles,``
           or the number of socket handles that could have been written if enough memory had been
           provided.
           If ``socket_handles`` is NULL, as output, ``socket_count`` will contain
           how many sockets are available to read in the system.

       socket_handles (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a block of memory to which the
           ::amdsmi_socket_handle values will be written. This value may be NULL.
           In this case, this function can be used to query how many sockets are
           available to read in the system.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_socket_handles(uint32_t * socket_count, amdsmi_socket_handle * socket_handles)


.. py:function:: amdsmi_get_socket_info(socket_handle, len, name)

   Get information about the given socket

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
   @platform{guest_mvf} @platform{guest_windows}

   This function retrieves socket information. The ``socket_handle`` must
   be provided to retrieve the Socket ID.

   Args:
       socket_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a socket handle

       len (:py:obj:`~.int`) -- *IN*:
           the length of the caller provided buffer ``name.``

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           The id of the socket.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_socket_info(amdsmi_socket_handle socket_handle, size_t len, char * name)


.. py:function:: amdsmi_get_processor_handles(socket_handle, processor_count, processor_handles)

   Get the list of the processor handles associated to a socket.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
   @platform{guest_mvf} @platform{guest_windows}

   This function retrieves the processor handles of a socket. The
   ``socket_handle`` must be provided for the processor. A socket may have multiple different
   type processors: An APU on a socket have both CPUs and GPUs.
   Currently, only AMD GPUs are supported.

   Note:
       Sockets are not supported on the @platform{host}.

   Note:
       On the @platform{host} this function currently supports only AMD GPUs. To enumerate other
       devices, such as AMD NICs, use amdsmi_get_processor_handles_by_type().

   The number of processor count is returned through ``processor_count``
   if ``processor_handles`` is NULL. Then the number of ``processor_count`` can be pass
   as input to retrieval all processors on the socket to ``processor_handles.``

   Args:
       socket_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The socket to query

       processor_count (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           As input, the value passed
           through this parameter is the number of ::amdsmi_processor_handle's that
           may be safely written to the memory pointed to by ``processor_handles.`` This is the
           limit on how many processor handles will be written to ``processor_handles.`` On return, `processor_count` will contain the number of processor handles written to ``processor_handles,``
           or the number of processor handles that could have been written if enough memory had been
           provided.
           If ``processor_handles`` is NULL, as output, ``processor_count`` will contain
           how many processors are available to read for the socket.

       processor_handles (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a block of memory to which the
           ::amdsmi_processor_handle values will be written. This value may be NULL.
           In this case, this function can be used to query how many processors are
           available to read.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle, uint32_t * processor_count, amdsmi_processor_handle * processor_handles)


.. py:function:: amdsmi_get_node_handle(processor_handle)

   Get the node handle associated with processor handle.

   @platform{gpu_bm_linux} @platform{host}

   This function retrieves the node handle of a processor handler. The
   ``processor_handle`` must be provided for the processor.
   Currently, only AMD GPUs are supported.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           A pointer to a ::amdsmi_processor_handle, this
           is required to be OAM ID 0 otherwise the API will fail. OAM ID is sourced
           from amdsmi_get_gpu_asic_info API.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               A pointer to a block of memory where amdsmi_node_handle
               will be written.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_node_handle(amdsmi_processor_handle processor_handle, amdsmi_node_handle * node_handle)


.. py:function:: amdsmi_get_processor_type(processor_handle)

   Get the processor type of the processor_handle

   @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf}
   @platform{guest_mvf} @platform{guest_windows}

   This function retrieves the processor type. A processor_handle must be provided
   for that processor.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_processor_type_t`:
               a pointer to ::amdsmi_processor_type_t to which the processor type
               will be written. If this parameter is nullptr, this function will return
               ::AMDSMI_STATUS_INVAL.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle, amdsmi_processor_type_t * processor_type)


.. py:function:: amdsmi_get_processor_info(processor_handle, len, name)

   Get a string identifier for the given processor.

   @platform{gpu_bm_linux} @platform{cpu_bm}

   This function writes the processor's index into ``name`` as a decimal
   string (for example "0", "1", "2"). The index is the processor's zero-based
   position in the library's processor list, the same order used by
   ::amdsmi_get_processor_handles. A valid ``processor_handle`` must be provided.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       len (:py:obj:`~.int`) -- *IN*:
           The length of the caller-provided buffer ``name.``

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           Buffer that receives the processor index as a decimal string.
           Must not be NULL.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_processor_info(amdsmi_processor_handle processor_handle, size_t len, char * name)


.. py:function:: amdsmi_get_processor_count_from_handles(processor_handles, processor_count)

   Get respective processor counts from the processor handles

   @platform{gpu_bm_linux} @platform{cpu_bm}

   This function classifies a list of processor handles and returns the per-type
   totals. Counts are derived purely from ::amdsmi_get_processor_type and do not require
   ENABLE_ESMI_LIB; on builds without ESMI, ``nr_cpusockets`` and ``nr_cpucores`` will be 0.

   Args:
       processor_handles (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           A pointer to a block of memory to which the
           ::amdsmi_processor_handle values will be written. This value may be NULL.

       processor_count (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN*:
           total processor count per socket

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               Total number of cpu sockets
       * :py:obj:`~.int`:
               Total number of cpu cores
       * :py:obj:`~.int`:
               Total number of gpu devices

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_processor_count_from_handles(amdsmi_processor_handle * processor_handles, uint32_t * processor_count, uint32_t * nr_cpusockets, uint32_t * nr_cpucores, uint32_t * nr_gpus)


.. py:function:: amdsmi_get_processor_handles_by_type(socket_handle, processor_type, processor_handles, processor_count)

   Returns a list of processor handles of the specified type in the system.

   @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm}

   This function retrieves processor list as per the processor type
   from the total processor handles list.
   The ``list`` of processor_handles and processor type must be provided.

   Note:
       This function fills the user-provided buffer with processor handles of the given type
       (e.g., GPU, NIC). The processor handles returned are used to instantiate the rest of processor
       queries in the library. If the buffer is not large enough, the call will fail.

   Args:
       socket_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The socket to query.

       processor_type (:py:obj:`~.amdsmi_processor_type_t`) -- *IN*:
           The type of processor to query (see ::amdsmi_processor_type_t).

       processor_handles (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *OUT*:
           Reference to list of processor handles returned by
           the library. Buffer must be allocated by user.

       processor_count (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           As input, the size of the provided buffer.
           As output, number of processor handles in the buffer.
           Parameter must be allocated by user.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_processor_handles_by_type(amdsmi_socket_handle socket_handle, amdsmi_processor_type_t processor_type, amdsmi_processor_handle * processor_handles, uint32_t * processor_count)


.. py:function:: amdsmi_get_processor_handle_from_bdf(bdf)

   Get processor handle with the matching bdf.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
   @platform{guest_mvf} @platform{guest_windows}

   Given bdf info ``bdf,`` this function will get
   the processor handle with the matching bdf.

   Args:
       bdf (:py:obj:`~.amdsmi_bdf_t`) -- *IN*:
           The bdf to match with corresponding processor handle.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               processor handle with the matching bdf.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_processor_handle_from_bdf(amdsmi_bdf_t bdf, amdsmi_processor_handle * processor_handle)


.. py:function:: amdsmi_get_gpu_device_bdf(processor_handle, bdf)

   Returns BDF of the given GPU device

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
   @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       bdf (:py:obj:`~.amdsmi_bdf_t`/:py:obj:`~.object`) -- *OUT*:
           Reference to BDF. Must be allocated by user.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_device_bdf(amdsmi_processor_handle processor_handle, amdsmi_bdf_t * bdf)


.. py:function:: amdsmi_get_gpu_device_uuid(processor_handle, uuid_length, uuid)

   Returns the UUID of the device

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
   @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       uuid_length (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           Length of the uuid string. As input, must be
           equal or greater than AMDSMI_GPU_UUID_SIZE and be allocated by
           user. As output it is the length of the uuid string.

       uuid (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           Pointer to string to store the UUID. Must be
           allocated by user.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_device_uuid(amdsmi_processor_handle processor_handle, unsigned int * uuid_length, char * uuid)


.. py:function:: amdsmi_get_gpu_enumeration_info(processor_handle)

   Returns the Enumeration information for the device

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_mvf}

   This function returns Enumeration information of the corresponding
   processor_handle. It will return the render number, card number,
   HSA ID, HIP ID, and the HIP UUID.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_enumeration_info_t`:
               Reference to Enumeration information structure.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_enumeration_info(amdsmi_processor_handle processor_handle, amdsmi_enumeration_info_t * info)


.. py:function:: amdsmi_get_cpu_affinity_with_scope(processor_handle, cpu_set_size, cpu_set, scope)

   Retrieves an array of uint64_t (sized to cpu_set_size) of bitmasks with the
    affinity within numa node or socket for the device.

   @platform{gpu_bm_linux} @platform{host}

   Given a processor handle ``processor_handle,`` the size of the cpu_set array `cpu_set_size`, and a pointer to an array of int64_t ``cpu_set,`` and ``scope,`` this function will
   write the CPU affinity bitmask to the array pointed to by ``cpu_set.``

   User must allocate the enough memory for the cpu_set array. The size of the array is determined
   by the number of CPU cores in the system. As an example, if there are 2 CPUs and each has 112
   cores, the size should be ceiling(2*112/64) = 4, where 64 is the bits of uint64_t. The function
   will write the CPU affinity bitmask to the array. For example, to describe the CPU cores
   0-55,112-167, it will set the 0-55 and 112-167 bits to 1 and the reset of bits to 0 in the
   cpu_set array.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       cpu_set_size (:py:obj:`~.int`) -- *IN*:
           The size of the cpu_set array that is safe to access

       cpu_set (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           Array reference in which to return a bitmask of CPU cores that this
           processor affinities with.

       scope (:py:obj:`~.amdsmi_affinity_scope_t`) -- *IN*:
           Scope for socket or numa affinity.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_affinity_with_scope(amdsmi_processor_handle processor_handle, uint32_t cpu_set_size, uint64_t * cpu_set, amdsmi_affinity_scope_t scope)


.. py:function:: amdsmi_get_gpu_virtualization_mode(processor_handle, mode)

   Returns the virtualization mode for the target device.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}

   The virtualization mode is detected and returned as an enum.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The identifier of the given device.

       mode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Reference to the enum representing virtualization mode.
           - When zero, the virtualization mode is unknown
           - When non-zero, the virtualization mode is detected

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_virtualization_mode(amdsmi_processor_handle processor_handle, amdsmi_virtualization_mode_t * mode)


.. py:function:: amdsmi_get_nic_processor_handles(socket_handle, processor_count, processor_handles)

   Get the list of the NIC processor handles associated to a socket.

   @platform{gpu_bm_linux} @platform{host}

   This function retrieves the processor handles of a socket. The
   ``socket_handle`` must be provided for the processor.

   Note:
       Sockets are not supported on the @platform{host}.

   The number of processor count is returned through ``processor_count``
   if ``processor_handles`` is NULL. Then the number of ``processor_count`` can be pass
   as input to retrieval all processors on the socket to ``processor_handles.``

   Args:
       socket_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The socket to query

       processor_count (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           As input, the value passed
           through this parameter is the number of ::amdsmi_processor_handle's that
           may be safely written to the memory pointed to by ``processor_handles.`` This is the
           limit on how many processor handles will be written to ``processor_handles.`` On return, `processor_count` will contain the number of processor handles written to ``processor_handles,``
           or the number of processor handles that could have been written if enough memory had been
           provided.
           If ``processor_handles`` is NULL, as output, ``processor_count`` will contain
           how many processors are available to read for the socket.

       processor_handles (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a block of memory to which the
           ::amdsmi_processor_handle values will be written. This value may be NULL.
           In this case, this function can be used to query how many processors are
           available to read.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_processor_handles(amdsmi_socket_handle socket_handle, uint32_t * processor_count, amdsmi_processor_handle * processor_handles)


.. py:function:: amdsmi_get_nic_device_bdf(processor_handle, bdf)

   Returns BDF of the given NIC device

   @platform{gpu_bm_linux} @platform{host}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       bdf (:py:obj:`~.amdsmi_bdf_t`/:py:obj:`~.object`) -- *OUT*:
           Reference to BDF. Must be allocated by user.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_device_bdf(amdsmi_processor_handle processor_handle, amdsmi_bdf_t * bdf)


.. py:function:: amdsmi_get_gpu_id(processor_handle, id)

   Get the device id associated with the device with provided device
   handler.

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint32_t ``id,``
   this function will write the device id value to the uint64_t pointed to by
   ``id.`` This ID is an identification of the type of device, so calling this
   function for different devices will give the same value if they are kind
   of device. Consequently, this function should not be used to distinguish
   one device from another. amdsmi_get_gpu_bdf_id() should be used to get a
   unique identifier.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       id (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the device id will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_id(amdsmi_processor_handle processor_handle, uint16_t * id)


.. py:function:: amdsmi_get_gpu_revision(processor_handle)

   Get the device revision associated with the device

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a
   uint16_t ``revision`` to which the revision id will be written

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               a pointer to uint16_t to which the device revision
               will be written

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_revision(amdsmi_processor_handle processor_handle, uint16_t * revision)


.. py:function:: amdsmi_get_gpu_vendor_name(processor_handle, name, len)

   Get the name string for a give vendor ID

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a pointer to a caller provided
   char buffer ``name,`` and a length of this buffer ``len,`` this function will
   write the name of the vendor (up to ``len`` characters) buffer ``name.`` The
   ``id`` may be a device vendor or subsystem vendor ID.

   If the integer ID associated with the vendor is not found in one of the
   system files containing device name information (e.g.
   /usr/share/misc/pci.ids), then this function will return the hex vendor ID
   as a string. Updating the system name files can be accompplished with
   "sudo update-pciids".

   Note:
       AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if ``len`` bytes is not
       large enough to hold the entire name. In this case, only ``len`` bytes will
       be written.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to a caller provided char buffer to which the
           name will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

       len (:py:obj:`~.int`) -- *IN*:
           the length of the caller provided buffer ``name.``

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_vendor_name(amdsmi_processor_handle processor_handle, char * name, size_t len)


.. py:function:: amdsmi_get_gpu_vram_vendor(processor_handle, brand, len)

   Get the vram vendor string of a device.

   Deprecated:
       This API is slated for removal in a future ROCm release;
       ::amdsmi_get_gpu_vram_info() should be used instead

   @platform{gpu_bm_linux}

   This function retrieves the vram vendor name given a processor handle
   ``processor_handle,`` a pointer to a caller provided
   char buffer ``brand,`` and a length of this buffer ``len,`` this function
   will write the vram vendor of the device (up to ``len`` characters) to the
   buffer ``brand.``

   If the vram vendor for the device is not found as one of the values
   contained within amdsmi_get_gpu_vram_vendor, then this function will return
   the string 'unknown' instead of the vram vendor.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       brand (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to a caller provided char buffer to which the
           vram vendor will be written

       len (:py:obj:`~.int`) -- *IN*:
           the length of the caller provided buffer ``brand.``

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_vram_vendor(amdsmi_processor_handle processor_handle, char * brand, uint32_t len)


.. py:function:: amdsmi_get_gpu_subsystem_id(processor_handle, id)

   Get the subsystem device id associated with the device with
   provided processor handle.

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint32_t ``id,``
   this function will write the subsystem device id value to the uint64_t
   pointed to by ``id.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       id (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the subsystem device id
           will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_subsystem_id(amdsmi_processor_handle processor_handle, uint16_t * id)


.. py:function:: amdsmi_get_gpu_subsystem_name(processor_handle, name, len)

   Get the name string for the device subsystem

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a pointer to a caller provided
   char buffer ``name,`` and a length of this buffer ``len,`` this function
   will write the name of the device subsystem (up to ``len`` characters)
   to the buffer ``name.``

   If the integer ID associated with the sub-system is not found in one of the
   system files containing device name information (e.g.
   /usr/share/misc/pci.ids), then this function will return the hex sub-system
   ID as a string. Updating the system name files can be accompplished with
   "sudo update-pciids".

   Note:
       AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if ``len`` bytes is not
       large enough to hold the entire name. In this case, only ``len`` bytes will
       be written.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to a caller provided char buffer to which the
           name will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

       len (:py:obj:`~.int`) -- *IN*:
           the length of the caller provided buffer ``name.``

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_subsystem_name(amdsmi_processor_handle processor_handle, char * name, size_t len)


.. py:function:: amdsmi_get_gpu_pci_bandwidth(processor_handle, bandwidth)

   Get the list of possible PCIe bandwidths that are available. It is not
   supported on virtual machine guest

   @platform{gpu_bm_linux} @platform{host}

   Given a processor handle ``processor_handle`` and a pointer to a to an
   ::amdsmi_pcie_bandwidth_t structure ``bandwidth,`` this function will fill in
   ``bandwidth`` with the possible T/s values and associated number of lanes,
   and indication of the current selection.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       bandwidth (:py:obj:`~.amdsmi_pcie_bandwidth_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to a caller provided
           ::amdsmi_pcie_bandwidth_t structure to which the frequency information will be
           written

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle, amdsmi_pcie_bandwidth_t * bandwidth)


.. py:function:: amdsmi_get_gpu_bdf_id(processor_handle, bdfid)

   Get the unique PCI device identifier associated for a device

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint64_t `bdfid`, this function will write the Bus/Device/Function PCI identifier
   (BDFID) associated with device ``processor_handle`` to the value pointed to by
   ``bdfid.``

   The format of ``bdfid`` will be as follows:

       BDFID = ((DOMAIN & 0xFFFFFFFF) << 32) | ((Partition & 0xF) << 28)
               | ((BUS & 0xFF) << 8) | ((DEVICE & 0x1F) <<3 )
               | (FUNCTION & 0x7)

   | Name         | Field   | KFD property     | KFD -> PCIe ID (uint64_t)    |
   -------------- | ------- | ---------------- | ---------------------------- |
   | Domain       | [63:32] | "domain"         | (DOMAIN & 0xFFFFFFFF) << 32  |
   | Partition id | [31:28] | "location id"    | (LOCATION & 0xF0000000)      |
   | Reserved     | [27:16] | "location id"    | N/A                          |
   | Bus          | [15: 8] | "location id"    | (LOCATION & 0xFF00)          |
   | Device       | [ 7: 3] | "location id"    | (LOCATION & 0xF8)            |
   | Function     | [ 2: 0] | "location id"    | (LOCATION & 0x7)             |

   Note: In some devices, the partition ID may be stored in the function bits
   BDFID[2:0] instead of BDFID[31:28].

   Note: For MI series devices, the function bits are only used to store the
   partition ID, but this modified BDF is internal to the ROCm stack.
   To the OS, partitions share the same BDF as the unpartitioned device and
   have function bits = 0, which can be verified through lspci.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       bdfid (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the device bdfid value
           will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_bdf_id(amdsmi_processor_handle processor_handle, uint64_t * bdfid)


.. py:function:: amdsmi_get_gpu_topo_numa_affinity(processor_handle, numa_node)

   Get the NUMA node associated with a device

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a int32_t `numa_node`, this function will retrieve the NUMA node value associated
   with device ``processor_handle`` and store the value at location pointed to by
   ``numa_node.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       numa_node (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN,OUT*:
           pointer to location where NUMA node value will
           be written.
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_topo_numa_affinity(amdsmi_processor_handle processor_handle, int32_t * numa_node)


.. py:function:: amdsmi_get_gpu_pci_throughput(processor_handle, sent, received, max_pkt_sz)

   Get PCIe traffic information. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Give a processor handle ``processor_handle`` and pointers to a uint64_t's, `sent`, ``received`` and ``max_pkt_sz,`` this function will write the number
   of bytes sent and received in 1 second to ``sent`` and ``received,``
   respectively. The maximum possible packet size will be written to
   ``max_pkt_sz.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sent (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the number of bytes sent
           will be written in 1 second. If pointer is NULL, it will be ignored.

       received (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the number of bytes
           received will be written. If pointer is NULL, it will be ignored.

       max_pkt_sz (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the maximum packet
           size will be written. If pointer is NULL, it will be ignored.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_pci_throughput(amdsmi_processor_handle processor_handle, uint64_t * sent, uint64_t * received, uint64_t * max_pkt_sz)


.. py:function:: amdsmi_get_gpu_pci_replay_counter(processor_handle, counter)

   Get PCIe replay counter

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint64_t `counter`, this function will write the sum of the number of NAK's received
   by the GPU and the NAK's generated by the GPU to memory pointed to by `counter`.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       counter (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the sum of the NAK's
           received and generated by the GPU is written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_pci_replay_counter(amdsmi_processor_handle processor_handle, uint64_t * counter)


.. py:function:: amdsmi_set_gpu_pci_bandwidth(processor_handle, bw_bitmask)

   Control the set of allowed PCIe bandwidths that can be used. It is not
   supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a 64 bit bitmask ``bw_bitmask,``
   this function will limit the set of allowable bandwidths. If a bit in `bw_bitmask` has a value of 1, then the frequency (as ordered in an
   ::amdsmi_frequencies_t returned by :: amdsmi_get_clk_freq()) corresponding
   to that bit index will be allowed.

   This function will change the performance level to
   ::AMDSMI_DEV_PERF_LEVEL_MANUAL in order to modify the set of allowable
   band_widths. Caller will need to set to ::AMDSMI_DEV_PERF_LEVEL_AUTO in order
   to get back to default state.

   All bits with indices greater than or equal to the value of the
   ::amdsmi_frequencies_t::num_supported field of ::amdsmi_pcie_bandwidth_t will be
   ignored.

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       bw_bitmask (:py:obj:`~.int`) -- *IN*:
           A bitmask indicating the indices of the
           bandwidths that are to be enabled (1) and disabled (0). Only the lowest
           ::amdsmi_frequencies_t::num_supported (of ::amdsmi_pcie_bandwidth_t) bits of
           this mask are relevant.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle, uint64_t bw_bitmask)


.. py:function:: amdsmi_get_energy_count(processor_handle, energy_accumulator, counter_resolution, timestamp)

   Get the energy accumulator counter of the processor with provided
   processor handle. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a pointer to a uint64_t
   ``energy_accumulator,`` and a pointer to a uint64_t ``timestamp,`` this function
   will write amount of energy consumed to the uint64_t pointed to by
   ``energy_accumulator,`` and the timestamp to the uint64_t pointed to by ``timestamp.``
   This function accumulates all energy consumed.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       energy_accumulator (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the energy
           counter will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

       counter_resolution (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           resolution of the counter ``energy_accumulator`` in
           micro Joules

       timestamp (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the timestamp
           will be written. Resolution: 1 ns.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_energy_count(amdsmi_processor_handle processor_handle, uint64_t * energy_accumulator, float * counter_resolution, uint64_t * timestamp)


.. py:function:: amdsmi_set_power_cap(processor_handle, sensor_ind, cap)

   Set the maximum gpu power cap value. It is not supported on virtual
   machine guest

   @platform{host} @platform{gpu_bm_linux} @platform{guest_1vf}

   Set the power cap to the provided value ``cap.``
   ``cap`` must be between the minimum and maximum power cap values set by the
   system, which can be obtained from ::amdsmi_dev_power_cap_range_get.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           A processor handle

       sensor_ind (:py:obj:`~.int`) -- *IN*:
           a 0-based sensor index. Normally, this will be 0.
           If a processor has more than one sensor, it could be greater than 0.

       cap (:py:obj:`~.int`) -- *IN*:
           a uint64_t that indicates the desired power cap.
           The ``cap`` value must be greater than 0.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_power_cap(amdsmi_processor_handle processor_handle, uint32_t sensor_ind, uint64_t cap)


.. py:function:: amdsmi_set_gpu_power_profile(processor_handle, reserved, profile)

   Set the power performance profile. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   This function will attempt to set the current profile to the provided
   profile, given a processor handle ``processor_handle`` and a ``profile.`` The provided
   profile must be one of the currently supported profiles, as indicated by a
   call to :: amdsmi_get_gpu_power_profile_presets()

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       reserved (:py:obj:`~.int`) -- *IN*:
           Not currently used. Set to 0.

       profile (:py:obj:`~.amdsmi_power_profile_preset_masks_t`) -- *IN*:
           a ::amdsmi_power_profile_preset_masks_t that hold the mask
           of the desired new power profile

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_power_profile(amdsmi_processor_handle processor_handle, uint32_t reserved, amdsmi_power_profile_preset_masks_t profile)


.. py:function:: amdsmi_get_supported_power_cap(processor_handle, sensor_inds, sensor_types)

   Query the supported power cap sensors and their types for a device.

   @platform{gpu_bm_linux} @platform{host}

   This function returns the list of supported power cap sensors for the given device,
   including their sensor indices and types (e.g., PPT0, PPT1).

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           A processor handle.

       sensor_inds (:py:obj:`~.rocm.bindings.util.types.ListOfUnsigned`/:py:obj:`~.object`) -- *OUT*:
           Pointer to an array of uint32_t to be filled with sensor indices.
           The array must be allocated by the caller with enough space.

       sensor_types (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to an array of amdsmi_power_cap_type_t to be filled with sensor
           types. The array must be allocated by the caller with enough space.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail.
       * :py:obj:`~.int`:
               Pointer to a uint32_t that will be set to the number of supported
               sensors.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_supported_power_cap(amdsmi_processor_handle processor_handle, uint32_t * sensor_count, uint32_t * sensor_inds, amdsmi_power_cap_type_t * sensor_types)


.. py:function:: amdsmi_get_cpu_socket_power(processor_handle)

   Get the socket power.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               - Input buffer to return socket power in milliwatts (mW)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_power(amdsmi_processor_handle processor_handle, uint32_t * ppower)


.. py:function:: amdsmi_get_cpu_socket_power_cap(processor_handle)

   Get the socket power cap.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               - Input buffer to return power cap in milliwatts (mW)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_power_cap(amdsmi_processor_handle processor_handle, uint32_t * pcap)


.. py:function:: amdsmi_get_cpu_socket_power_cap_max(processor_handle)

   Get the maximum power cap value for a given socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               - Input buffer to return maximum power limit value in milliwatts (mW)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_power_cap_max(amdsmi_processor_handle processor_handle, uint32_t * pmax)


.. py:function:: amdsmi_get_cpu_pwr_svi_telemetry_all_rails(processor_handle, power)

   Get the SVI based power telemetry for all rails.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       power (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return svi based power value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_pwr_svi_telemetry_all_rails(amdsmi_processor_handle processor_handle, uint32_t * power)


.. py:function:: amdsmi_set_cpu_socket_power_cap(processor_handle, pcap)

   Set the power cap value for a given socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       pcap (:py:obj:`~.int`) -- *IN*:
           - Input power limit value in milliwatts (mW)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_socket_power_cap(amdsmi_processor_handle processor_handle, uint32_t pcap)


.. py:function:: amdsmi_set_cpu_pwr_efficiency_mode(processor_handle, power_efficiency_mode, utilization, ppt_limit)

   Set the power efficiency profile policy.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to configure

       power_efficiency_mode (:py:obj:`~.int`) -- *IN*:
           - power efficiency mode to be set

       utilization (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - pointer to store utilization for balanced core modes (%)

       ppt_limit (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - pointer to PPT (Package Power Tracking) limit value
           in milliwatts (mW)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_pwr_efficiency_mode(amdsmi_processor_handle processor_handle, uint8_t power_efficiency_mode, uint32_t * utilization, uint32_t * ppt_limit)


.. py:function:: amdsmi_get_cpu_pwr_efficiency_mode(processor_handle)

   Get the power efficiency profile policy

   This function retrieves the current power efficiency mode, utility value,
   and PPT (Package Power Tracking) limit for a given processor socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               - pointer to store current power efficiency mode
       * :py:obj:`~.int`:
               - pointer to store utilization for balanced core modes (%)
       * :py:obj:`~.int`:
               - pointer to store PPT (Package Power Tracking) limit value
               in milliwatts (mW)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_pwr_efficiency_mode(amdsmi_processor_handle processor_handle, uint32_t * power_efficiency_mode, uint32_t * utilization, uint32_t * ppt_limit)


.. py:function:: amdsmi_get_cpu_core_ccd_power(processor_handle)

   Read CCD (Core Complex Die) power consumption

   This function reads the power consumption of a specific CCD within a CPU socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu core which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on successful register read, non-zero on failure
       * :py:obj:`~.int`:
               - Input buffer to store power consumption in milliwatts (mW)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_core_ccd_power(amdsmi_processor_handle processor_handle, uint32_t * power)


.. py:function:: amdsmi_get_gpu_memory_total(processor_handle, mem_type, total)

   Get the total amount of memory that exists

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a type of memory ``mem_type,`` and
   a pointer to a uint64_t ``total,`` this function will write the total amount
   of ``mem_type`` memory that exists to the location pointed to by ``total.``

   Note:
       Sum of the process memory is not expected to be the total memory usage.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       mem_type (:py:obj:`~.amdsmi_memory_type_t`) -- *IN*:
           The type of memory for which the total amount will be
           found

       total (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the total amount of
           memory will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_memory_total(amdsmi_processor_handle processor_handle, amdsmi_memory_type_t mem_type, uint64_t * total)


.. py:function:: amdsmi_get_gpu_memory_usage(processor_handle, mem_type, used)

   Get the current memory usage

   @platform{gpu_bm_linux}

   This function will write the amount of ``mem_type`` memory that
   that is currently being used to the location pointed to by ``used.``

   Note:
       Sum of the process memory is not expected to be the total memory usage.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       mem_type (:py:obj:`~.amdsmi_memory_type_t`) -- *IN*:
           The type of memory for which the amount being used will
           be found

       used (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint64_t to which the amount of memory
           currently being used will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_memory_usage(amdsmi_processor_handle processor_handle, amdsmi_memory_type_t mem_type, uint64_t * used)


.. py:function:: amdsmi_get_gpu_bad_page_info(processor_handle)

   Get the bad pages of a processor. It is not supported on virtual
   machine guest

   @platform{gpu_bm_linux}

   This call will query the device ``processor_handle`` for the
   number of bad pages (written to ``num_pages`` address). The results are
   written to address held by the ``info`` pointer.
   The first call to this API returns the number of bad pages which
   should be used to allocate the buffer that should contain the bad page
   records.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               Number of bad page records.
       * :py:obj:`~.amdsmi_retired_page_record_t`:
               The results will be written to the
               amdsmi_retired_page_record_t pointer.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_bad_page_info(amdsmi_processor_handle processor_handle, uint32_t * num_pages, amdsmi_retired_page_record_t * info)


.. py:function:: amdsmi_get_gpu_bad_page_threshold(processor_handle)

   Get the bad pages threshold of a processor. It is not supported on virtual
   machine guest

   @platform{gpu_bm_linux}

   This call will query the device ``processor_handle`` for the
   threshold of bad pages (written to ``threshold`` address).

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               of bad page count.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_bad_page_threshold(amdsmi_processor_handle processor_handle, uint32_t * threshold)


.. py:function:: amdsmi_gpu_validate_ras_eeprom(processor_handle)

   Verify the checksum of RAS EEPROM. It is not supported on virtual
   machine guest

   @platform{gpu_bm_linux}

   This call will verify the device ``processor_handle`` for the
   checksum of RAS EEPROM.

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success
           AMDSMI_STATUS_CORRUPTED_EEPROM on the device's EEPROM corruption
           others on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_gpu_validate_ras_eeprom(amdsmi_processor_handle processor_handle)


.. py:function:: amdsmi_get_gpu_ras_block_features_enabled(processor_handle, block, state)

   Returns if RAS features are enabled or disabled for given block. It is not
   supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` this function queries the
   state of RAS features for a specific block ``block.`` Result will be written
   to address held by pointer ``state.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device handle which to query

       block (:py:obj:`~.amdsmi_gpu_block_t`) -- *IN*:
           Block which to query

       state (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to amdsmi_ras_err_state_t to which the state
           of block will be written.
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_ras_block_features_enabled(amdsmi_processor_handle processor_handle, amdsmi_gpu_block_t block, amdsmi_ras_err_state_t * state)


.. py:function:: amdsmi_get_gpu_memory_reserved_pages(processor_handle, num_pages, records)

   Get information about reserved ("retired") memory pages. It is not supported on
   virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` this function returns retired page
   information ``records`` corresponding to the device with the provided processor
   handle ``processor_handle.`` The number of retired page records is returned through `num_pages`. ``records`` may be NULL on input. In this case, the number of
   records available for retrieval will be returned through ``num_pages.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       num_pages (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to a uint32. As input, the value passed
           through this parameter is the number of ::amdsmi_retired_page_record_t's that
           may be safely written to the memory pointed to by ``records.`` This is the
           limit on how many records will be written to ``records.`` On return, `num_pages` will contain the number of records written to ``records,`` or the
           number of records that could have been written if enough memory had been
           provided.
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

       records (:py:obj:`~.amdsmi_retired_page_record_t`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a block of memory to which the
           ::amdsmi_retired_page_record_t values will be written. This value may be NULL.
           In this case, this function can be used to query how many records are
           available to read.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_memory_reserved_pages(amdsmi_processor_handle processor_handle, uint32_t * num_pages, amdsmi_retired_page_record_t * records)


.. py:function:: amdsmi_get_gpu_fan_rpms(processor_handle, sensor_ind, speed)

   Get the fan speed in RPMs of the device with the specified processor
   handle and 0-based sensor index. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint32_t
   ``speed,`` this function will write the current fan speed in RPMs to the
   uint32_t pointed to by ``speed``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sensor_ind (:py:obj:`~.int`) -- *IN*:
           a 0-based sensor index. Normally, this will be 0.
           If a device has more than one sensor, it could be greater than 0.

       speed (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint32_t to which the speed will be
           written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_fan_rpms(amdsmi_processor_handle processor_handle, uint32_t sensor_ind, int64_t * speed)


.. py:function:: amdsmi_get_gpu_fan_speed(processor_handle, sensor_ind, speed)

   Get the fan speed for the specified device as a value relative to
   the maximum fan speed. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint32_t
   ``speed,`` this function will write the current fan speed (a value
   between 0 and the maximum fan speed) to the uint32_t pointed to by ``speed.``
   For legacy hwmon GPUs the maximum is ::AMDSMI_MAX_FAN_SPEED (255).
   For GPUs with the gpu_od sysfs interface, use amdsmi_get_gpu_fan_speed_max()
   to query the actual maximum

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sensor_ind (:py:obj:`~.int`) -- *IN*:
           a 0-based sensor index. Normally, this will be 0.
           If a device has more than one sensor, it could be greater than 0.

       speed (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint32_t to which the speed will be
           written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_fan_speed(amdsmi_processor_handle processor_handle, uint32_t sensor_ind, int64_t * speed)


.. py:function:: amdsmi_get_gpu_fan_speed_max(processor_handle, sensor_ind, max_speed)

   Get the max. fan speed of the device with provided processor handle. It is
   not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint32_t
   ``max_speed,`` this function will write the maximum fan speed possible to
   the uint32_t pointed to by ``max_speed.``
   For legacy hwmon GPUs this is ::AMDSMI_MAX_FAN_SPEED (255).
   For GPUs with the gpu_od sysfs interface, the maximum is read from the
   OD_RANGE section of the fan_minimum_pwm sysfs file (e.g. 100)

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sensor_ind (:py:obj:`~.int`) -- *IN*:
           a 0-based sensor index. Normally, this will be 0.
           If a device has more than one sensor, it could be greater than 0.

       max_speed (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint32_t to which the maximum speed
           will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_fan_speed_max(amdsmi_processor_handle processor_handle, uint32_t sensor_ind, uint64_t * max_speed)


.. py:function:: amdsmi_get_gpu_cache_info(processor_handle)

   Returns gpu cache info.

   @platform{gpu_bm_linux} @platform{host}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           PF of a processor for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_gpu_cache_info_t`:
               reference to the cache info struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_cache_info(amdsmi_processor_handle processor_handle, amdsmi_gpu_cache_info_t * info)


.. py:function:: amdsmi_get_gpu_volt_metric(processor_handle, sensor_type, metric, voltage)

   Get the voltage metric value for the specified metric, from the
   specified voltage sensor on the specified device. It is not supported on
   virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a sensor type ``sensor_type,`` a
   ::amdsmi_voltage_metric_t ``metric`` and a pointer to an int64_t `voltage`, this function will write the value of the metric indicated by
   ``metric`` and ``sensor_type`` to the memory location ``voltage.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sensor_type (:py:obj:`~.amdsmi_voltage_type_t`) -- *IN*:
           part of device from which voltage should be
           obtained. This should come from the enum ::amdsmi_voltage_type_t

       metric (:py:obj:`~.amdsmi_voltage_metric_t`) -- *IN*:
           enum indicated which voltage value should be
           retrieved

       voltage (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to int64_t to which the voltage
           will be written, in millivolts.
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle, amdsmi_voltage_type_t sensor_type, amdsmi_voltage_metric_t metric, int64_t * voltage)


.. py:function:: amdsmi_reset_gpu_fan(processor_handle, sensor_ind)

   Reset the fan to automatic driver control. It is not supported on virtual
   machine guest

   @platform{gpu_bm_linux}

   This function returns control of the fan to the system.
   For GPUs with the gpu_od sysfs interface, this writes the OD_RANGE minimum
   value to fan_minimum_pwm and commits the change

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sensor_ind (:py:obj:`~.int`) -- *IN*:
           a 0-based sensor index. Normally, this will be 0.
           If a device has more than one sensor, it could be greater than 0.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_reset_gpu_fan(amdsmi_processor_handle processor_handle, uint32_t sensor_ind)


.. py:function:: amdsmi_set_gpu_fan_speed(processor_handle, sensor_ind, speed)

   Set the fan speed for the specified device with the provided speed,
   in RPMs. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a integer value indicating
   speed ``speed,`` this function will attempt to set the fan speed to ``speed.``
   An error will be returned if the specified speed is outside the allowable
   range for the device. For legacy hwmon GPUs the range is 0-255.
   For GPUs with the gpu_od sysfs interface, the valid range is determined
   dynamically from the OD_RANGE (e.g. 20-100).

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sensor_ind (:py:obj:`~.int`) -- *IN*:
           a 0-based sensor index. Normally, this will be 0.
           If a device has more than one sensor, it could be greater than 0.

       speed (:py:obj:`~.int`) -- *IN*:
           the speed to which the function will attempt to set the fan

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_fan_speed(amdsmi_processor_handle processor_handle, uint32_t sensor_ind, uint64_t speed)


.. py:function:: amdsmi_get_gpu_busy_percent(processor_handle, gpu_busy_percent)

   Get GPU busy percent from gpu_busy_percent sysfs file

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` this function returns GPU busy
   percentage.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       gpu_busy_percent (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           Direct output from the gpu_busy_percent sysfs file

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_busy_percent(amdsmi_processor_handle processor_handle, uint32_t * gpu_busy_percent)


.. py:function:: amdsmi_get_vcn_busy_percent(processor_handle)

   Get VCN busy percent from vcn_busy_percent sysfs file

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` this function returns VCN busy
   percentage.

   @retval ::AMDSMI_STATUS_SUCCESS on success
    @retval ::AMDSMI_STATUS_NOT_SUPPORTED if the device does not support this query
    @retval ::AMDSMI_STATUS_INVAL if the input parameters are invalid
    @retval ::AMDSMI_STATUS_UNEXPECTED_DATA if data read from the sysfs file is not in the expected
   format or empty

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
       * :py:obj:`~.int`:
               vcn busy percentage (0-100)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_vcn_busy_percent(amdsmi_processor_handle processor_handle, uint32_t * vcn_busy_percent)


.. py:function:: amdsmi_get_utilization_count(processor_handle, utilization_counters, count, timestamp)

   Get coarse grain utilization counter of the specified device

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` the array of the utilization counters,
   the size of the array, this function returns the coarse grain utilization counters
   and timestamp.
   The counter is the accumulated percentages. Every milliseconds the firmware calculates
   % busy count and then accumulates that value in the counter. This provides minimally
   invasive coarse grain GPU usage information.

   If the function returns AMDSMI_STATUS_SUCCESS, the counter will be set in the value field of
   the amdsmi_utilization_counter_t.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       utilization_counters (:py:obj:`~.amdsmi_utilization_counter_t`/:py:obj:`~.object`) -- *IN,OUT*:
           Multiple utilization counters can be retrieved with a single
           call. The caller must allocate enough space to the utilization_counters array. The caller also
           needs to set valid AMDSMI_UTILIZATION_COUNTER_TYPE type for each element of the array.
           ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the provided arguments.

       count (:py:obj:`~.int`) -- *IN*:
           The size of ``utilization_counters`` array.

       timestamp (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           The timestamp when the counter is retrieved. Resolution: 1 ns.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_utilization_count(amdsmi_processor_handle processor_handle, amdsmi_utilization_counter_t[] utilization_counters, uint32_t count, uint64_t * timestamp)


.. py:function:: amdsmi_get_gpu_perf_level(processor_handle, perf)

   Get the performance level of the device. It is not supported on virtual
   machine guest

   @platform{gpu_bm_linux}

   This function will write the ::amdsmi_dev_perf_level_t to the uint32_t
   pointed to by ``perf,`` for a given processor handle ``processor_handle`` and a pointer
   to a uint32_t ``perf.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       perf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to ::amdsmi_dev_perf_level_t to which the
           performance level will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_perf_level(amdsmi_processor_handle processor_handle, amdsmi_dev_perf_level_t * perf)


.. py:function:: amdsmi_set_gpu_perf_determinism_mode(processor_handle, clkvalue)

   Enter performance determinism mode with provided processor handle. It is
   not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and ``clkvalue`` this function
   will enable performance determinism mode, which enforces a GFXCLK frequency
   SoftMax limit per GPU set by the user. This prevents the GFXCLK PLL from
   stretching when running the same workload on different GPUS, making
   performance variation minimal. This call will result in the performance
   level ::amdsmi_dev_perf_level_t of the device being
   ::AMDSMI_DEV_PERF_LEVEL_DETERMINISM.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       clkvalue (:py:obj:`~.int`) -- *IN*:
           Softmax value for GFXCLK in MHz.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_perf_determinism_mode(amdsmi_processor_handle processor_handle, uint64_t clkvalue)


.. py:function:: amdsmi_get_gpu_overdrive_level(processor_handle, od)

   Get the overdrive percent associated with the device with provided
   processor handle. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint32_t ``od,``
   this function will write the overdrive percentage to the uint32_t pointed
   to by ``od``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       od (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint32_t to which the overdrive percentage
           will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t * od)


.. py:function:: amdsmi_get_gpu_mem_overdrive_level(processor_handle, od)

   Get the GPU memory clock overdrive percent associated with the device with provided
   processor handle. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a uint32_t ``od,``
   this function will write the overdrive percentage to the uint32_t pointed
   to by ``od``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       od (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to uint32_t to which the GPU memory clock overdrive percentage
           will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_mem_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t * od)


.. py:function:: amdsmi_get_clk_freq(processor_handle, clk_type, f)

   Get the list of possible system clock speeds of device for a
   specified clock type. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a clock type ``clk_type,`` and a
   pointer to a to an ::amdsmi_frequencies_t structure ``f,`` this function will
   fill in ``f`` with the possible clock speeds, and indication of the current
   clock speed selection.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       clk_type (:py:obj:`~.amdsmi_clk_type_t`) -- *IN*:
           the type of clock for which the frequency is desired

       f (:py:obj:`~.amdsmi_frequencies_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to a caller provided ::amdsmi_frequencies_t structure
           to which the frequency information will be written. Frequency values are in
           Hz.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_clk_freq(amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_frequencies_t * f)


.. py:function:: amdsmi_reset_gpu(processor_handle)

   Triggers a chain that resets all GPUs.
   It is not supported on virtual machine guest

   @platform{gpu_bm_linux} @platform{host}

   Note:
       After this function returns, the caller must wait a few seconds before calling
       any other AMD SMI API functions to allow the GPU reset to complete. Calling other APIs
       too soon may result in AMDSMI_STATUS_BUSY or undefined behavior.

   Given a processor handle ``processor_handle,`` this function will reset the GPU

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle processor_handle)


.. py:function:: amdsmi_get_gpu_od_volt_info(processor_handle, odv)

   This function retrieves the overdrive GFX & MCLK information. If valid
   for the GPU it will also populate the voltage curve data. It is not supported
   on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a
   ::amdsmi_od_volt_freq_data_t structure ``odv,`` this function will populate `odv`. See ::amdsmi_od_volt_freq_data_t for more details.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       odv (:py:obj:`~.amdsmi_od_volt_freq_data_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to an ::amdsmi_od_volt_freq_data_t structure
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_od_volt_info(amdsmi_processor_handle processor_handle, amdsmi_od_volt_freq_data_t * odv)


.. py:function:: amdsmi_get_gpu_metrics_header_info(processor_handle, header_value)

   Get the 'metrics_header_info' from the GPU metrics associated with the device

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a
   amd_metrics_table_header_t in which the 'metrics_header_info' will stored

   @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
   ::AMDSMI_STATUS_NOT_SUPPORTED is returned in case the metric unit
     does not exist for the given device

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       header_value (:py:obj:`~.amd_metrics_table_header_t`/:py:obj:`~.object`) -- *INOUT*:
           a pointer to amd_metrics_table_header_t to which the device gpu
           metric unit will be stored

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_metrics_header_info(amdsmi_processor_handle processor_handle, amd_metrics_table_header_t * header_value)


.. py:function:: amdsmi_get_gpu_metrics_info(processor_handle, pgpu_metrics)

   This function retrieves the gpu metrics information. It is not supported
   on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a
   ::amdsmi_gpu_metrics_t structure ``pgpu_metrics,`` this function will populate
   ``pgpu_metrics.`` See ::amdsmi_gpu_metrics_t for more details.

   **APU Metrics:**
   When APU-specific metrics are available (APU metrics table v2.4 or v3.0),
   ``pgpu_metrics->apu_metrics`` will point to thread-local library-owned storage.
   This pointer is invalidated by ANY subsequent call to
   ::amdsmi_get_gpu_metrics_info or ::amdsmi_get_gpu_partition_metrics_info on
   the same thread, even for different devices. Callers must copy the entire
   ::amdsmi_apu_metrics_t structure immediately after the call to preserve the data.
   For non-APU devices, ``pgpu_metrics->apu_metrics`` will be nullptr.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       pgpu_metrics (:py:obj:`~.amdsmi_gpu_metrics_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to an ::amdsmi_gpu_metrics_t structure
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_metrics_info(amdsmi_processor_handle processor_handle, amdsmi_gpu_metrics_t * pgpu_metrics)


.. py:function:: amdsmi_get_gpu_partition_metrics_info(processor_handle, pgpu_metrics)

   This function retrieves the partition metrics information.

   @platform{gpu_bm_linux} @platform{guest_1vf}

   Given a processor handle ``processor_handle`` and a pointer to a
   ::amdsmi_gpu_metrics_t structure ``pgpu_metrics,`` this function will populate
   ``pgpu_metrics.`` See ::amdsmi_gpu_metrics_t for more details.

   **APU Metrics:**
   When APU-specific metrics are available (APU metrics table v2.4 or v3.0),
   ``pgpu_metrics->apu_metrics`` will point to thread-local library-owned storage.
   This pointer is invalidated by ANY subsequent call to
   ::amdsmi_get_gpu_metrics_info or ::amdsmi_get_gpu_partition_metrics_info on
   the same thread, even for different devices. Callers must copy the entire
   ::amdsmi_apu_metrics_t structure immediately after the call to preserve the data.
   For non-APU devices, ``pgpu_metrics->apu_metrics`` will be nullptr.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       pgpu_metrics (:py:obj:`~.amdsmi_gpu_metrics_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to an ::amdsmi_gpu_metrics_t structure
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_partition_metrics_info(amdsmi_processor_handle processor_handle, amdsmi_gpu_metrics_t * pgpu_metrics)


.. py:function:: amdsmi_get_gpu_pm_metrics_info(processor_handle, pm_metrics, num_of_metrics)

   Get the pm metrics table with provided device index.

   @platform{gpu_bm_linux}

   Given a device handle ``processor_handle,`` ``pm_metrics`` pointer,
   and ``num_of_metrics`` pointer,
   this function will write the pm metrics name value pair
   to the array at ``pm_metrics`` and the number of metrics retrieved to ``num_of_metrics``
   Note: the library allocated memory for pm_metrics, and user must call
   free(pm_metrics) to free it after use.

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function with the given arguments
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       pm_metrics (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *INOUT*:
           A pointerto an array to hold multiple PM metrics. On success,
           the library will allocate memory of pm_metrics and write metrics to this array.
           The caller must free this memory after usage to avoid memory leak.

       num_of_metrics (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *INOUT*:
           a pointer to uint32_t to which the number of
           metrics is allocated for pm_metrics array as input, and the number of metrics retrieved
           as output. If this parameter is NULL, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_pm_metrics_info(amdsmi_processor_handle processor_handle, amdsmi_name_value_t ** pm_metrics, uint32_t * num_of_metrics)


.. py:function:: amdsmi_get_gpu_reg_table_info(processor_handle, reg_type, reg_metrics, num_of_metrics)

   Get the register metrics table with provided device index and register type.

   @platform{gpu_bm_linux}

   Given a device handle ``processor_handle,`` ``reg_type,`` ``reg_metrics`` pointer,
   and ``num_of_metrics`` pointer,
   this function will write the register metrics name value pair
   to the array at ``reg_metrics`` and the number of metrics retrieved to ``num_of_metrics``
   Note: the library allocated memory for reg_metrics, and user must call
   free(reg_metrics) to free it after use.

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function with the given arguments
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       reg_type (:py:obj:`~.amdsmi_reg_type_t`) -- *IN*:
           The register type

       reg_metrics (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *INOUT*:
           A pointerto an array to hold multiple register metrics. On success,
           the library will allocate memory of reg_metrics and write metrics to this array.
           The caller must free this memory after usage to avoid memory leak.

       num_of_metrics (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *INOUT*:
           a pointer to uint32_t to which the number of
           metrics is allocated for reg_metrics array as input, and the number of metrics retrieved
           as output. If this parameter is NULL, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_reg_table_info(amdsmi_processor_handle processor_handle, amdsmi_reg_type_t reg_type, amdsmi_name_value_t ** reg_metrics, uint32_t * num_of_metrics)


.. py:function:: amdsmi_set_gpu_clk_limit(processor_handle, clk_type, limit_type, clk_value)

   This function sets the clock sets the clock min/max level

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a clock type ``clk_type,``
   a value ``clk_value`` needs to be set, and the ``level`` indicates min or max
   clock you want to set, this function the clock limit.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       clk_type (:py:obj:`~.amdsmi_clk_type_t`) -- *IN*:
           AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_TYPE_MEM and so on

       limit_type (:py:obj:`~.amdsmi_clk_limit_type_t`) -- *IN*:
           AMDSMI_FREQ_IND_MIN|AMDSMI_FREQ_IND_MAX to set the
           minimum (0) or maximum (1) speed.

       clk_value (:py:obj:`~.int`) -- *IN*:
           value to apply to. Frequency values are in MHz.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_limit_type_t limit_type, uint64_t clk_value)


.. py:function:: amdsmi_set_gpu_od_clk_info(processor_handle, level, clkvalue, clkType)

   This function sets the clock frequency information. It is not supported on
   virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a frequency level ``level,``
   a clock value ``clkvalue`` and a clock type ``clkType`` this function
   will set the sclk|mclk range

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       level (:py:obj:`~.amdsmi_freq_ind_t`) -- *IN*:
           AMDSMI_FREQ_IND_MIN|AMDSMI_FREQ_IND_MAX to set the
           minimum (0) or maximum (1) speed.

       clkvalue (:py:obj:`~.int`) -- *IN*:
           value to apply to the clock range. Frequency values
           are in MHz.

       clkType (:py:obj:`~.amdsmi_clk_type_t`) -- *IN*:
           AMDSMI_CLK_TYPE_SYS | AMDSMI_CLK_TYPE_MEM range type

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_od_clk_info(amdsmi_processor_handle processor_handle, amdsmi_freq_ind_t level, uint64_t clkvalue, amdsmi_clk_type_t clkType)


.. py:function:: amdsmi_set_gpu_od_volt_info(processor_handle, vpoint, clkvalue, voltvalue)

   This function sets  1 of the 3 voltage curve points. It is not supported
   on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a voltage point ``vpoint``
   and a voltage value ``voltvalue`` this function will set voltage curve point

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       vpoint (:py:obj:`~.int`) -- *IN*:
           voltage point [0|1|2] on the voltage curve

       clkvalue (:py:obj:`~.int`) -- *IN*:
           clock value component of voltage curve point.
           Frequency values are in MHz.

       voltvalue (:py:obj:`~.int`) -- *IN*:
           voltage value component of voltage curve point.
           Voltage is in mV.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_od_volt_info(amdsmi_processor_handle processor_handle, uint32_t vpoint, uint64_t clkvalue, uint64_t voltvalue)


.. py:function:: amdsmi_get_gpu_od_volt_curve_regions(processor_handle, num_regions, buffer)

   This function will retrieve the current valid regions in the
   frequency/voltage space. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a pointer to an unsigned integer
   ``num_regions`` and a buffer of ::amdsmi_freq_volt_region_t structures, `buffer`, this function will populate ``buffer`` with the current
   frequency-volt space regions. The caller should assign ``buffer`` to memory
   that can be written to by this function. The caller should also
   indicate the number of ::amdsmi_freq_volt_region_t structures that can safely
   be written to ``buffer`` in ``num_regions.``

   The number of regions to expect this function provide (``num_regions)`` can
   be obtained by calling :: amdsmi_get_gpu_od_volt_info().

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       num_regions (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           As input, this is the number of
           ::amdsmi_freq_volt_region_t structures that can be written to ``buffer.`` As
           output, this is the number of ::amdsmi_freq_volt_region_t structures that were
           actually written.
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

       buffer (:py:obj:`~.amdsmi_freq_volt_region_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a caller provided buffer to which
           ::amdsmi_freq_volt_region_t structures will be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_od_volt_curve_regions(amdsmi_processor_handle processor_handle, uint32_t * num_regions, amdsmi_freq_volt_region_t * buffer)


.. py:function:: amdsmi_get_gpu_power_profile_presets(processor_handle, sensor_ind, status)

   Get the list of available preset power profiles and an indication of
   which profile is currently active. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer to a
   ::amdsmi_power_profile_status_t ``status,`` this function will set the bits of
   the ::amdsmi_power_profile_status_t.available_profiles bit field of ``status`` to
   1 if the profile corresponding to the respective
   ::amdsmi_power_profile_preset_masks_t profiles are enabled. For example, if both
   the VIDEO and VR power profiles are available selections, then
   ::AMDSMI_PWR_PROF_PRST_VIDEO_MASK AND'ed with
   ::amdsmi_power_profile_status_t.available_profiles will be non-zero as will
   ::AMDSMI_PWR_PROF_PRST_VR_MASK AND'ed with
   ::amdsmi_power_profile_status_t.available_profiles. Additionally,
   ::amdsmi_power_profile_status_t.current will be set to the
   ::amdsmi_power_profile_preset_masks_t of the profile that is currently active.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sensor_ind (:py:obj:`~.int`) -- *IN*:
           a 0-based sensor index. Normally, this will be 0.
           If a device has more than one sensor, it could be greater than 0.

       status (:py:obj:`~.amdsmi_power_profile_status_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to ::amdsmi_power_profile_status_t that will be
           populated by a call to this function
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_power_profile_presets(amdsmi_processor_handle processor_handle, uint32_t sensor_ind, amdsmi_power_profile_status_t * status)


.. py:function:: amdsmi_set_gpu_perf_level(processor_handle, perf_lvl)

   Set the PowerPlay performance level associated with the device with
   provided processor handle with the provided value. It is not supported
   on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and an ::amdsmi_dev_perf_level_t `perf_level`, this function will set the PowerPlay performance level for the
   device to the value ``perf_lvl.``

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       perf_lvl (:py:obj:`~.amdsmi_dev_perf_level_t`) -- *IN*:
           the value to which the performance level should be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_perf_level(amdsmi_processor_handle processor_handle, amdsmi_dev_perf_level_t perf_lvl)


.. py:function:: amdsmi_set_gpu_overdrive_level(processor_handle, od)

   Set the overdrive percent associated with the device with provided
   processor handle with the provided value. See details for WARNING. It is
   not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and an overdrive level ``od,``
   this function will set the overdrive level for the device to the value
   ``od.`` The overdrive level is an integer value between 0 and 20, inclusive,
   which represents the overdrive percentage; e.g., a value of 5 specifies
   an overclocking of 5%.

   The overdrive level is specific to the gpu system clock.

   The overdrive level is the percentage above the maximum Performance Level
   to which overclocking will be limited. The overclocking percentage does
   not apply to clock speeds other than the maximum. This percentage is
   limited to 20%.

    ******WARNING******
   Operating your AMD GPU outside of official AMD specifications or outside of
   factory settings, including but not limited to the conducting of
   overclocking (including use of this overclocking software, even if such
   software has been directly or indirectly provided by AMD or otherwise
   affiliated in any way with AMD), may cause damage to your AMD GPU, system
   components and/or result in system failure, as well as cause other problems.
   DAMAGES CAUSED BY USE OF YOUR AMD GPU OUTSIDE OF OFFICIAL AMD SPECIFICATIONS
   OR OUTSIDE OF FACTORY SETTINGS ARE NOT COVERED UNDER ANY AMD PRODUCT
   WARRANTY AND MAY NOT BE COVERED BY YOUR BOARD OR SYSTEM MANUFACTURER'S
   WARRANTY. Please use this utility with caution.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       od (:py:obj:`~.int`) -- *IN*:
           the value to which the overdrive level should be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t od)


.. py:function:: amdsmi_set_clk_freq(processor_handle, clk_type, freq_bitmask)

   Control the set of allowed frequencies that can be used for the
   specified clock. It is not supported on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a clock type ``clk_type,`` and a
   64 bit bitmask ``freq_bitmask,`` this function will limit the set of
   allowable frequencies. If a bit in ``freq_bitmask`` has a value of 1, then
   the frequency (as ordered in an ::amdsmi_frequencies_t returned by
   amdsmi_get_clk_freq()) corresponding to that bit index will be
   allowed.

   This function will change the performance level to
   ::AMDSMI_DEV_PERF_LEVEL_MANUAL in order to modify the set of allowable
   frequencies. Caller will need to set to ::AMDSMI_DEV_PERF_LEVEL_AUTO in order
   to get back to default state.

   All bits with indices greater than or equal to
   ::amdsmi_frequencies_t::num_supported will be ignored.

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       clk_type (:py:obj:`~.amdsmi_clk_type_t`) -- *IN*:
           the type of clock for which the set of frequencies
           will be modified

       freq_bitmask (:py:obj:`~.int`) -- *IN*:
           A bitmask indicating the indices of the
           frequencies that are to be enabled (1) and disabled (0). Only the lowest
           ::amdsmi_frequencies_t.num_supported bits of this mask are relevant.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_clk_freq(amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, uint64_t freq_bitmask)


.. py:function:: amdsmi_get_soc_pstate(processor_handle, policy)

   Get the soc pstate policy for the processor

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{host}

   Given a processor handle ``processor_handle,`` this function will write
   current soc pstate  policy settings to ``policy.`` All the processors at the same socket
   will have the same policy.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       policy (:py:obj:`~.amdsmi_dpm_policy_t`/:py:obj:`~.object`) -- *IN,OUT*:
           the soc pstate policy for this processor.
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_soc_pstate(amdsmi_processor_handle processor_handle, amdsmi_dpm_policy_t * policy)


.. py:function:: amdsmi_set_soc_pstate(processor_handle, policy_id)

   Set the soc pstate policy for the processor

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{host}

   Given a processor handle ``processor_handle`` and a soc pstate  policy ``policy_id,``
   this function will set the soc pstate  policy for this processor. All the processors at
   the same socket will be set to the same policy.

   Note:
       This function requires admin/sudo privileges on @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       policy_id (:py:obj:`~.int`) -- *IN*:
           the soc pstate  policy id to set. The id is the id in
           amdsmi_dpm_policy_entry_t, which can be obtained by calling
           amdsmi_get_soc_pstate()

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_soc_pstate(amdsmi_processor_handle processor_handle, uint32_t policy_id)


.. py:function:: amdsmi_get_xgmi_plpd(processor_handle, xgmi_plpd)

   Get the xgmi per-link power down policy parameter for the processor

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{host}

   Given a processor handle ``processor_handle,`` this function will write
   current xgmi plpd settings to ``policy.`` All the processors at the same socket
   will have the same policy.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       xgmi_plpd (:py:obj:`~.amdsmi_dpm_policy_t`/:py:obj:`~.object`) -- *IN,OUT*:
           the xgmi plpd for this processor.
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_xgmi_plpd(amdsmi_processor_handle processor_handle, amdsmi_dpm_policy_t * xgmi_plpd)


.. py:function:: amdsmi_set_xgmi_plpd(processor_handle, policy_id)

   Set the xgmi per-link power down policy parameter for the processor

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{host}

   Given a processor handle ``processor_handle`` and a dpm policy ``policy_id,``
   this function will set the xgmi plpd for this processor. All the processors at
   the same socket will be set to the same policy.

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       policy_id (:py:obj:`~.int`) -- *IN*:
           the xgmi plpd id to set. The id is the id in
           amdsmi_dpm_policy_entry_t, which can be obtained by calling
           amdsmi_get_xgmi_plpd()

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_xgmi_plpd(amdsmi_processor_handle processor_handle, uint32_t policy_id)


.. py:function:: amdsmi_get_gpu_process_isolation(processor_handle, pisolate)

   Get the status of the Process Isolation

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_windows}

   Given a processor handle ``processor_handle,`` this function will write
   current process isolation status to ``pisolate.`` The 0 is the process isolation
   disabled, and the 1 is the process isolation enabled.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       pisolate (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           the process isolation status.
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_process_isolation(amdsmi_processor_handle processor_handle, uint32_t * pisolate)


.. py:function:: amdsmi_set_gpu_process_isolation(processor_handle, pisolate)

   Enable/disable the system Process Isolation

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_windows}

   Given a processor handle ``processor_handle`` and a process isolation ``pisolate,``
   flag, this function will set the Process Isolation for this processor. The 0 is the process
   isolation disabled, and the 1 is the process isolation enabled.

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       pisolate (:py:obj:`~.int`) -- *IN*:
           the process isolation status to set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_process_isolation(amdsmi_processor_handle processor_handle, uint32_t pisolate)


.. py:function:: amdsmi_clean_gpu_local_data(processor_handle)

   Run the cleaner shader to clean up data in LDS/GPRs

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_windows}

   Given a processor handle ``processor_handle,``
   this function will clean the local data of this processor. This can be called between
   user logins to prevent information leak.

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_clean_gpu_local_data(amdsmi_processor_handle processor_handle)


.. py:class:: amdsmi_fabric_telemetry_category_t

   Bases: :py:obj:`enum.IntEnum`


   Fabric telemetry categories
       


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_SWITCH
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_CRYPTO
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_PFC
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_DERIVED_UALOE
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_DERIVED_NETPORT
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_INVALID
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_UNKNOWN
      :type:  int


.. py:class:: amdsmi_fabric_telemetry_category_mask_t

   Bases: :py:obj:`enum.IntEnum`


   Fabric telemetry category bitmask values
       


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_UALOE
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_SWITCH
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_CRYPTO
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_PFC
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_NETPORT
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_UALOE
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_NETPORT
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_ALL_KNOWN
      :type:  int


.. py:class:: amdsmi_fabric_telemetry_item_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Fabric telemetry item structure
       


   .. py:attribute:: id
      :type:  Any


   .. py:attribute:: value
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_fabric_label_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: text
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_fabric_telemetry_instance_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Fabric telemetry instance structure

   Collection of telemetry data items for an instance of a category of telemetry


   .. py:attribute:: name
      :type:  Any


   .. py:attribute:: logical_idx
      :type:  Any


   .. py:attribute:: item_count
      :type:  Any


   .. py:attribute:: items
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_fabric_telemetry_dataset_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Fabric telemetry dataset structure

   Contains all telemetry for one category


   .. py:attribute:: category
      :type:  Any


   .. py:attribute:: generation_count
      :type:  Any


   .. py:attribute:: timestamp
      :type:  Any


   .. py:attribute:: instance_count
      :type:  Any


   .. py:attribute:: instances
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_fabric_telemetry_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Fabric telemetry structure

   Top level structure defining telemetry data for Fabric. Contains datasets
   for each category of telemetry. A null pointer means no telemetry is
   available for that category.


.. py:function:: amdsmi_alloc_fabric_telemetry(processor_handle, category_mask)

   Allocate storage for Fabric telemetry data

   @platform{gpu_bm_linux} @platform{host}

   This function allocates storage for Fabric telemetry data for the
   specified categories. The allocated storage can be reused for multiple
   telemetry retrievals.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Handle for the target processor

       category_mask (:py:obj:`~.int`) -- *IN*:
           - Bitmask of telemetry categories to allocate,
           constructed using AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK(cat)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_fabric_telemetry_t`:
               - Pointer to allocated telemetry structure

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_alloc_fabric_telemetry(amdsmi_processor_handle processor_handle, uint32_t category_mask, amdsmi_fabric_telemetry_t ** telemetry)


.. py:function:: amdsmi_get_fabric_telemetry_data(processor_handle, telemetry)

   Get Fabric telemetry data

   @platform{gpu_bm_linux} @platform{host}

   This function retrieves the latest Fabric telemetry data snapshot
   into pre-allocated storage.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Handle for the target processor

       telemetry (:py:obj:`~.amdsmi_fabric_telemetry_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Pre-allocated telemetry structure to populate

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_fabric_telemetry_data(amdsmi_processor_handle processor_handle, amdsmi_fabric_telemetry_t * telemetry)


.. py:function:: amdsmi_fabric_telem_id_to_string(telem_id, telem_name)

   Get string name for a telemetry item ID

   @platform{gpu_bm_linux}

   Given a telemetry item ID ``telem_id,``
   this function returns a pointer to a string containing the human-readable name
   for the specified telemetry item. The returned string is statically allocated
   and should not be freed by the caller.

   Args:
       telem_id (:py:obj:`~.int`) -- *IN*:
           The telemetry item ID for which the name is requested

       telem_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           The telemetry item name

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_fabric_telem_id_to_string(uint64_t telem_id, const char ** telem_name)


.. py:function:: amdsmi_free_fabric_telemetry(processor_handle, telemetry)

   Free Fabric telemetry storage

   @platform{gpu_bm_linux} @platform{host}

   This function frees the storage allocated for Fabric telemetry data.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Handle for the target processor

       telemetry (:py:obj:`~.amdsmi_fabric_telemetry_t`/:py:obj:`~.object`) -- *IN*:
           - Telemetry structure to free

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_free_fabric_telemetry(amdsmi_processor_handle processor_handle, amdsmi_fabric_telemetry_t * telemetry)


.. py:class:: amdsmi_fabric_size_constants_t

   Bases: :py:obj:`enum.IntEnum`


   Fabric size constants
       


   .. py:attribute:: AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_MAX_LOCAL_GPUS
      :type:  int


.. py:class:: amdsmi_fabric_type_t

   Bases: :py:obj:`enum.IntEnum`


   Fabric type
       


   .. py:attribute:: AMDSMI_FABRIC_TYPE_UALOE
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TYPE_UALINK
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TYPE_UNKNOWN
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_TYPE_UALLINK
      :type:  int


.. py:class:: amdsmi_fabric_npa_address_mode_t

   Bases: :py:obj:`enum.IntEnum`


   Fabric NPA address mode
       


   .. py:attribute:: AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_IDENTIFICATION
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN
      :type:  int


.. py:class:: amdsmi_fabric_accelerator_vpod_state_t

   Bases: :py:obj:`enum.IntEnum`


   Fabric accelerator vPoD state
       


   .. py:attribute:: AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNCONFIGURED
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR
      :type:  int


   .. py:attribute:: AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNKNOWN
      :type:  int


.. py:class:: amdsmi_fabric_info_v1_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Fabric device configuration information (version 1)
       


   .. py:attribute:: accelerator_id
      :type:  Any


   .. py:attribute:: fabric_type
      :type:  Any


   .. py:attribute:: bandwidth
      :type:  Any


   .. py:attribute:: latency
      :type:  Any


   .. py:attribute:: ppod_id
      :type:  Any


   .. py:attribute:: ppod_size
      :type:  Any


   .. py:attribute:: vpod_id
      :type:  Any


   .. py:attribute:: vpod_size
      :type:  Any


   .. py:attribute:: vpod_active_accelerators
      :type:  Any


   .. py:attribute:: local_accelerators
      :type:  Any


   .. py:attribute:: addr_mode
      :type:  Any


   .. py:attribute:: accel_state
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_fabric_info_t_fabric_info_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: v1
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_fabric_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Fabric device information structure
       


   .. py:attribute:: bdf
      :type:  Any


   .. py:attribute:: fabric_version
      :type:  Any


   .. py:attribute:: fabric_info
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: amdsmi_get_gpu_fabric_info(processor_handle)

   Get Fabric device information

   @platform{gpu_bm_linux} @platform{host}

   Reads optional UALoE fabric attributes from sysfs (one file per field).
   Missing or unreadable files are skipped so the call can return partial data:
     - any field that was not updated from sysfs keeps its sentinel value (ie:
       numeric fields at their maximum representable value, and unknown enumeration
       values where documented for ::amdsmi_fabric_info_v1_t).
     - The device BDF in ``info`` is always filled when the call completes successfully
       or returns ::AMDSMI_STATUS_NO_DATA.

   Note:
       This path reads sysfs only. It does not require UALoE netlink
       (::ualoe_open) to succeed; that handle is still needed for fabric telemetry APIs.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Handle for the target processor

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           - ::AMDSMI_STATUS_SUCCESS if at least one sysfs file yielded usable content.
           - ::AMDSMI_STATUS_NO_DATA if no sysfs file yielded usable lines (output still
             contains BDF and default/sentinel fabric fields).
           - Other codes (e.g. invalid processor handle) on failure.
       * :py:obj:`~.amdsmi_fabric_info_t`:
               - Pointer to Fabric information structure to be populated.
               Must be allocated by the caller. Written on every return except errors such
               as ::AMDSMI_STATUS_INVAL.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_fabric_info(amdsmi_processor_handle processor_handle, amdsmi_fabric_info_t * info)


.. py:function:: amdsmi_get_lib_version()

   Get the build version information for the currently running build of AMDSMI

   @platform{gpu_bm_linux} @platform{cpu_bm} @platform{guest_1vf} @platform{guest_mvf}
   @platform{guest_windows}

   Get the major, minor, patch and build string for AMDSMI build
   currently in use through ``version``

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_version_t`:
               A pointer to an ::amdsmi_version_t structure that will
               be updated with the version information upon return.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t * version)


.. py:function:: amdsmi_get_gpu_ecc_count(processor_handle, block, ec)

   Retrieve the error counts for a GPU block. It is not supported on virtual
   machine guest

   See [RAS Error Count sysfs Interface (AMDGPU RAS Support - Linux Kernel
   documentation)](https://docs.kernel.org/gpu/amdgpu/ras.html:py:obj:`~.ras`-error-count-sysfs-interface)
   to learn how these error counts are accessed.

   @platform{gpu_bm_linux} @platform{host}

   Given a processor handle ``processor_handle,`` an ::amdsmi_gpu_block_t ``block`` and a
   pointer to an ::amdsmi_error_count_t ``ec,`` this function will write the error
   count values for the GPU block indicated by ``block`` to memory pointed to by
   ``ec.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       block (:py:obj:`~.amdsmi_gpu_block_t`) -- *IN*:
           The block for which error counts should be retrieved

       ec (:py:obj:`~.amdsmi_error_count_t`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an ::amdsmi_error_count_t to which the error
           counts should be written
           If this parameter is nullptr, this function will return ::AMDSMI_STATUS_INVAL
           if the function is supported with the provided arguments and ::AMDSMI_STATUS_NOT_SUPPORTED
           if it is not supported with the provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_ecc_count(amdsmi_processor_handle processor_handle, amdsmi_gpu_block_t block, amdsmi_error_count_t * ec)


.. py:function:: amdsmi_get_gpu_ecc_enabled(processor_handle, enabled_blocks)

   Retrieve the enabled ECC bit-mask. It is not supported on virtual machine guest

   See [RAS Error Count sysfs Interface (AMDGPU RAS Support - Linux Kernel
   documentation)](https://docs.kernel.org/gpu/amdgpu/ras.html:py:obj:`~.ras`-error-count-sysfs-interface)
   to learn how these error counts are accessed.

   @platform{gpu_bm_linux} @platform{host}

   Given a processor handle ``processor_handle,`` and a pointer to a uint64_t `enabled_mask`, this function will write bits to memory pointed to by
   ``enabled_blocks.`` Upon a successful call, ``enabled_blocks`` can then be
   AND'd with elements of the ::amdsmi_gpu_block_t ennumeration to determine if
   the corresponding block has ECC enabled.

   Note:
       Whether a block has ECC enabled or not in the device is independent
       of whether there is kernel support for error counting for that block.
       Although a block may be enabled, but there may not be kernel support for
       reading error counters for that block.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       enabled_blocks (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a uint64_t to which the enabled
           blocks bits will be written.
           If this parameter is nullptr, this function will return ::AMDSMI_STATUS_INVAL
           if the function is supported with the provided arguments and ::AMDSMI_STATUS_NOT_SUPPORTED
           if it is not supported with the provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_ecc_enabled(amdsmi_processor_handle processor_handle, uint64_t * enabled_blocks)


.. py:function:: amdsmi_get_gpu_total_ecc_count(processor_handle)

   Returns the total number of ECC errors (correctable,
          uncorrectable and deferred) in the given GPU. It is not supported on
          virtual machine guest

   See [RAS Error Count sysfs Interface (AMDGPU RAS Support - Linux Kernel
   documentation)](https://docs.kernel.org/gpu/amdgpu/ras.html:py:obj:`~.ras`-error-count-sysfs-interface)
   to learn how these error counts are accessed.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_error_count_t`:
               Reference to ecc error count structure.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_total_ecc_count(amdsmi_processor_handle processor_handle, amdsmi_error_count_t * ec)


.. py:class:: amdsmi_cper_guid_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Cper
       


   .. py:attribute:: b
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_cper_timestamp_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: seconds
      :type:  Any


   .. py:attribute:: minutes
      :type:  Any


   .. py:attribute:: hours
      :type:  Any


   .. py:attribute:: flag
      :type:  Any


   .. py:attribute:: day
      :type:  Any


   .. py:attribute:: month
      :type:  Any


   .. py:attribute:: year
      :type:  Any


   .. py:attribute:: century
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_cper_valid_bits_t_valid_bits_(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: platform_id
      :type:  Any


   .. py:attribute:: timestamp
      :type:  Any


   .. py:attribute:: partition_id
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_cper_valid_bits_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: valid_bits
      :type:  Any


   .. py:attribute:: valid_mask
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_cper_hdr_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: signature
      :type:  Any


   .. py:attribute:: revision
      :type:  Any


   .. py:attribute:: signature_end
      :type:  Any


   .. py:attribute:: sec_cnt
      :type:  Any


   .. py:attribute:: error_severity
      :type:  Any


   .. py:attribute:: cper_valid_bits
      :type:  Any


   .. py:attribute:: record_length
      :type:  Any


   .. py:attribute:: timestamp
      :type:  Any


   .. py:attribute:: platform_id
      :type:  Any


   .. py:attribute:: partition_id
      :type:  Any


   .. py:attribute:: creator_id
      :type:  Any


   .. py:attribute:: notify_type
      :type:  Any


   .. py:attribute:: record_id
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:attribute:: persistence_info
      :type:  Any


   .. py:attribute:: reserved
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: amdsmi_get_afids_from_cper(cper_buffer, buf_size, afids, num_afids)

   Get the AFIDs from CPER buffer

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
   @platform{guest_mvf}

   A utility function which retrieves the AFIDs from the CPER record.

   Args:
       cper_buffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a pointer to the buffer with one CPER record.
           The caller must make sure the whole CPER record is loaded into the buffer.

       buf_size (:py:obj:`~.int`) -- *IN*:
           is the size of the cper_buffer.

       afids (:py:obj:`~.rocm.bindings.util.types.ListOfUInt64`/:py:obj:`~.object`) -- *OUT*:
           a pointer to an array of uint64_t to which the AF IDs will be written

       num_afids (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           As input, the value passed through this parameter is the number of
            uint64_t that may be safely written to the memory pointed to by ``afids.`` This is the limit
            on how many AF IDs will be written to ``afids.`` On return, ``num_afids`` will contain the
            number of AF IDs written to ``afids,`` or the number of AF IDs that could have been written
            if enough memory had been provided. It is suggest to pass AMDSMI_MAX_NUMBER_OF_AFIDS_PER_RECORD
           for all AF Ids.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_afids_from_cper(char * cper_buffer, uint32_t buf_size, uint64_t * afids, uint32_t * num_afids)


.. py:function:: amdsmi_get_gpu_ras_feature_info(processor_handle)

   Returns RAS features info.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device handle which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_ras_feature_t`:
               RAS features that are currently enabled and supported on
               the processor. Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_ras_feature_info(amdsmi_processor_handle processor_handle, amdsmi_ras_feature_t * ras_feature)


.. py:function:: amdsmi_get_gpu_cper_entries(processor_handle, severity_mask, cper_data, buf_size, cper_hdrs, entry_count, cursor)

   Retrieve CPER entries cached in the driver.

   The user will pass buffers to hold the CPER data and CPER headers. The library will
   fill the buffer based on the severity_mask user passed. It will also parse the CPER header
   and stored in the cper_hdrs array. The user can use the cper_hdrs to get the timestamp and other
   header information. A cursor is also returned to the user, which can be used to get the next set
   of CPER entries.

   If there are more data than any of the buffers user pass, the library will return
   AMDSMI_STATUS_MORE_DATA. User can call the API again with the cursor returned at previous call to
   get more data. If the buffer size is too small to even hold one entry, the library will return
   AMDSMI_STATUS_OUT_OF_RESOURCES.

   Even if the API returns AMDSMI_STATUS_MORE_DATA, the 2nd call may still get the entry_count == 0
   as the driver cache may not contain the severity user is interested in. The API returns
   AMDSMI_STATUS_SUCCESS with entry_count == 0 and buf_size == 0 in this case so that user can
   ignore that call.

   An empty CPER ring (no records) also returns AMDSMI_STATUS_SUCCESS with
   entry_count == 0 and buf_size == 0.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Handle to the processor for which CPER entries are to be retrieved.

       severity_mask (:py:obj:`~.int`) -- *IN*:
           The severity mask of the entries to be retrieved.

       cper_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Pointer to a buffer where the CPER data will be stored. User must
           allocate the buffer and set the buf_size correctly.

       buf_size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           Pointer to a variable that specifies the size of the cper_data.
           On return, it will contain the actual size of the data written to the cper_data.

       cper_hdrs (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Array of the parsed headers of the cper_data. The user must allocate
                          the array of pointers to cper_hdr. The library will fill the array with the
           pointers to the parsed headers. The underlying data is in the cper_data buffer and only pointer
           is stored in this array.

       entry_count (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           Pointer to a variable that specifies the array length of the cper_hdrs
           user allocated. On return, it will contain the actual entries written to the cper_hdrs.

       cursor (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           Pointer to a variable that will contain the  cursor  for the next call.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_cper_entries(amdsmi_processor_handle processor_handle, uint32_t severity_mask, char * cper_data, uint64_t * buf_size, amdsmi_cper_hdr_t ** cper_hdrs, uint64_t * entry_count, uint64_t * cursor)


.. py:function:: amdsmi_get_gpu_ecc_status(processor_handle, block, state)

   Retrieve the ECC status for a GPU block. It is not supported on virtual machine
   guest

   See [RAS Error Count sysfs Interface (AMDGPU RAS Support - Linux Kernel
   documentation)](https://docs.kernel.org/gpu/amdgpu/ras.html:py:obj:`~.ras`-error-count-sysfs-interface)
   to learn how these error counts are accessed.

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` an ::amdsmi_gpu_block_t ``block`` and
   a pointer to an ::amdsmi_ras_err_state_t ``state,`` this function will write
   the current state for the GPU block indicated by ``block`` to memory pointed
   to by ``state.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       block (:py:obj:`~.amdsmi_gpu_block_t`) -- *IN*:
           The block for which error counts should be retrieved

       state (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an ::amdsmi_ras_err_state_t to which the
           ECC state should be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_ecc_status(amdsmi_processor_handle processor_handle, amdsmi_gpu_block_t block, amdsmi_ras_err_state_t * state)


.. py:function:: amdsmi_status_code_to_string(status, status_string)

   Get a description of a provided AMDSMI error status

   @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm}
   @platform{guest_1vf} @platform{guest_mvf} @platform{guest_windows}

   Set the provided pointer to a const char *, ``status_string,`` to
   a string containing a description of the provided error code ``status.``

   Args:
       status (:py:obj:`~.amdsmi_status_t`) -- *IN*:
           The error status for which a description is desired

       status_string (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a const char * which will be made
           to point to a description of the provided error code

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_status_code_to_string(amdsmi_status_t status, const char ** status_string)


.. py:function:: amdsmi_gpu_counter_group_supported(processor_handle, group)

   Tell if an event group is supported by a given device. It is not supported
   on virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and an event group specifier `group`, tell if ``group`` type events are supported by the device associated
   with ``processor_handle``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           processor handle of device being queried

       group (:py:obj:`~.amdsmi_event_group_t`) -- *IN*:
           amdsmi_event_group_t identifier of group for which support
           is being queried

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_gpu_counter_group_supported(amdsmi_processor_handle processor_handle, amdsmi_event_group_t group)


.. py:function:: amdsmi_gpu_create_counter(processor_handle, type, evnt_handle)

   Create a performance counter object

   @platform{gpu_bm_linux}

   Create a performance counter object of type ``type`` for the device
   with a processor handle of ``processor_handle,`` and write a handle to the object to the
   memory location pointed to by ``evnt_handle.`` ``evnt_handle`` can be used
   with other performance event operations. The handle should be deallocated
   with ::amdsmi_gpu_destroy_counter() when no longer needed.

   Note:
       This function requires admin/sudo privileges

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       type (:py:obj:`~.amdsmi_event_type_t`) -- *IN*:
           the ::amdsmi_event_type_t of performance event to create

       evnt_handle (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a ::amdsmi_event_handle_t which will be
           associated with a newly allocated counter
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_gpu_create_counter(amdsmi_processor_handle processor_handle, amdsmi_event_type_t type, amdsmi_event_handle_t * evnt_handle)


.. py:function:: amdsmi_gpu_destroy_counter(evnt_handle)

   Deallocate a performance counter object

   @platform{gpu_bm_linux}

   Deallocate the performance counter object with the provided
   ::amdsmi_event_handle_t ``evnt_handle``

   Note:
       This function requires admin/sudo privileges

   Args:
       evnt_handle (:py:obj:`~.int`) -- *IN*:
           handle to event object to be deallocated

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_gpu_destroy_counter(amdsmi_event_handle_t evnt_handle)


.. py:function:: amdsmi_gpu_control_counter(evt_handle, cmd, cmd_args)

   Issue performance counter control commands. It is not supported on
   virtual machine guest

   @platform{gpu_bm_linux}

   Issue a command ``cmd`` on the event counter associated with the
   provided handle ``evt_handle.``

   Note:
       This function requires admin/sudo privileges

   Args:
       evt_handle (:py:obj:`~.int`) -- *IN*:
           an event handle

       cmd (:py:obj:`~.amdsmi_counter_command_t`) -- *IN*:
           The event counter command to be issued

       cmd_args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           Currently not used. Should be set to NULL.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_gpu_control_counter(amdsmi_event_handle_t evt_handle, amdsmi_counter_command_t cmd, void * cmd_args)


.. py:function:: amdsmi_gpu_read_counter(evt_handle, value)

   Read the current value of a performance counter

   @platform{gpu_bm_linux}

   Read the current counter value of the counter associated with the
   provided handle ``evt_handle`` and write the value to the location pointed
   to by ``value.``

   Note:
       This function requires admin/sudo privileges

   Args:
       evt_handle (:py:obj:`~.int`) -- *IN*:
           an event handle

       value (:py:obj:`~.amdsmi_counter_value_t`/:py:obj:`~.object`) -- *IN,OUT*:
           pointer to memory of size of ::amdsmi_counter_value_t to
           which the counter value will be written

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle, amdsmi_counter_value_t * value)


.. py:function:: amdsmi_get_gpu_available_counters(processor_handle, grp, available)

   Get the number of currently available counters. It is not supported on
   virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a performance event group ``grp,``
   and a pointer to a uint32_t ``available,`` this function will write the
   number of ``grp`` type counters that are available on the device with handle
   ``processor_handle`` to the memory that ``available`` points to.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       grp (:py:obj:`~.amdsmi_event_group_t`) -- *IN*:
           an event device group

       available (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a uint32_t to which the number of
           available counters will be written

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_available_counters(amdsmi_processor_handle processor_handle, amdsmi_event_group_t grp, uint32_t * available)


.. py:function:: amdsmi_get_gpu_compute_process_info(procs, num_items)

   Get process information about processes currently using GPU

   @platform{gpu_bm_linux}

   Given a non-NULL pointer to an array ``procs`` of
   ::amdsmi_process_info_t's, of length *``num_items,`` this function will write
   up to *``num_items`` instances of ::amdsmi_process_info_t to the memory pointed
   to by ``procs.`` These instances contain information about each process
   utilizing a GPU. If ``procs`` is not NULL, ``num_items`` will be updated with
   the number of processes actually written. If ``procs`` is NULL, ``num_items``
   will be updated with the number of processes for which there is current
   process information. Calling this function with ``procs`` being NULL is a way
   to determine how much memory should be allocated for when ``procs`` is not
   NULL.

   Args:
       procs (:py:obj:`~.amdsmi_process_info_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to memory provided by the caller to which
           process information will be written. This may be NULL in which case only `num_items` will be updated with the number of processes found.

       num_items (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a uint32_t, which on input, should
           contain the amount of memory in ::amdsmi_process_info_t's which have been
           provided by the ``procs`` argument. On output, if ``procs`` is non-NULL, this
           will be updated with the number ::amdsmi_process_info_t structs actually
           written. If ``procs`` is NULL, this argument will be updated with the number
           processes for which there is information.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_compute_process_info(amdsmi_process_info_t * procs, uint32_t * num_items)


.. py:function:: amdsmi_get_gpu_compute_process_info_by_pid(pid, proc)

   Get process information about a specific process

   @platform{gpu_bm_linux}

   Given a pointer to an ::amdsmi_process_info_t ``proc`` and a process
   id
   ``pid,`` this function will write the process information for ``pid,`` if
   available, to the memory pointed to by ``proc.``

   Args:
       pid (:py:obj:`~.int`) -- *IN*:
           The process ID for which process information is being
           requested

       proc (:py:obj:`~.amdsmi_process_info_t`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to a ::amdsmi_process_info_t to which
           process information for ``pid`` will be written if it is found.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_compute_process_info_by_pid(uint32_t pid, amdsmi_process_info_t * proc)


.. py:function:: amdsmi_get_gpu_compute_process_gpus(pid, dv_indices, num_devices)

   Get the device indices currently being used by a process

   @platform{gpu_bm_linux}

   Given a process id ``pid,`` a non-NULL pointer to an array of
   uint32_t's ``processor_handleices`` of length *``num_devices,`` this function will
   write up to ``num_devices`` device indices to the memory pointed to by
   ``processor_handleices.`` If ``processor_handleices`` is not NULL, ``num_devices`` will be
   updated with the number of gpu's currently being used by process ``pid.``
   If ``processor_handleices`` is NULL, ``processor_handleices`` will be updated with the number of
   gpus currently being used by ``pid.`` Calling this function with `dv_indices` being NULL is a way to determine how much memory is required
   for when ``processor_handleices`` is not NULL.

   Args:
       pid (:py:obj:`~.int`) -- *IN*:
           The process id of the process for which the number of gpus
           currently being used is requested

       dv_indices (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to memory provided by the caller to
           which indices of devices currently being used by the process will be
           written. This may be NULL in which case only ``num_devices`` will be
           updated with the number of devices being used.

       num_devices (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a uint32_t, which on input, should
           contain the amount of memory in uint32_t's which have been provided by the
           ``processor_handleices`` argument. On output, if ``processor_handleices`` is non-NULL, this will
           be updated with the number uint32_t's actually written. If ``processor_handleices`` is
           NULL, this argument will be updated with the number devices being used.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t * dv_indices, uint32_t * num_devices)


.. py:function:: amdsmi_gpu_xgmi_error_status(processor_handle, status)

   Retrieve the XGMI error status for a device. It is not supported on
   virtual machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` and a pointer to an
   ::amdsmi_xgmi_status_t ``status,`` this function will write the current XGMI
   error state ::amdsmi_xgmi_status_t for the device ``processor_handle`` to the memory
   pointed to by ``status.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       status (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an ::amdsmi_xgmi_status_t to which the
           XGMI error state should be written
           If this parameter is nullptr, this function will return
           ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
           arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
           provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_gpu_xgmi_error_status(amdsmi_processor_handle processor_handle, amdsmi_xgmi_status_t * status)


.. py:function:: amdsmi_reset_gpu_xgmi_error(processor_handle)

   Reset the XGMI error status for a device. It is not supported on virtual
   machine guest

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` this function will reset the
   current XGMI error state ::amdsmi_xgmi_status_t for the device ``processor_handle`` to
   amdsmi_xgmi_status_t::AMDSMI_XGMI_STATUS_NO_ERRORS

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_reset_gpu_xgmi_error(amdsmi_processor_handle processor_handle)


.. py:function:: amdsmi_get_xgmi_info(processor_handle)

   Returns XGMI information for the GPU.

   @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_xgmi_info_t`:
               Reference to xgmi information structure. Must be
               allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_xgmi_info(amdsmi_processor_handle processor_handle, amdsmi_xgmi_info_t * info)


.. py:function:: amdsmi_get_gpu_xgmi_link_status(processor_handle)

   Get the XGMI link status

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,``  this function
   will return the link status for each XGMI link connect to this processor.
   If the processor link type is not XGMI, it should return AMDSMI_STATUS_NOT_SUPPORTED.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_xgmi_link_status_t`:
               The link status of the XGMI connect to this processor.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle processor_handle, amdsmi_xgmi_link_status_t * link_status)


.. py:function:: amdsmi_get_link_metrics(processor_handle)

   Return link metric information

   @platform{gpu_bm_linux} @platform{host}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           PF of a processor for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_link_metrics_t`:
               reference to the link metrics struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_link_metrics(amdsmi_processor_handle processor_handle, amdsmi_link_metrics_t * link_metrics)


.. py:function:: amdsmi_topo_get_numa_node_number(processor_handle, numa_node)

   Retrieve the NUMA CPU node number for a device

   @platform{gpu_bm_linux} @platform{host}

   Given a processor handle ``processor_handle,`` and a pointer to an
   uint32_t ``numa_node,`` this function will write the
   node number of NUMA CPU for the device ``processor_handle`` to the memory
   pointed to by ``numa_node.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       numa_node (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an uint32_t to which the
           numa node number should be written.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_topo_get_numa_node_number(amdsmi_processor_handle processor_handle, uint32_t * numa_node)


.. py:function:: amdsmi_topo_get_link_weight(processor_handle_src, processor_handle_dst, weight)

   Retrieve the weight for a connection between 2 GPUs

   @platform{gpu_bm_linux}

   Given a source processor handle ``processor_handle_src`` and
   a destination processor handle ``processor_handle_dst,`` and a pointer to an
   uint64_t ``weight,`` this function will write the
   weight for the connection between the device ``processor_handle_src``
   and ``processor_handle_dst`` to the memory pointed to by ``weight.``

   The weight is a qualitative cost metric derived from the KFD io_link
   ``weight`` property (lower values indicate closer or faster connections),
   similar in spirit to the NUMA distances reported by ``numactl.`` The value
   is computed as follows:

   - Each physical xGMI hop contributes 15, so an xGMI route traversing
     *N* physical links has a weight of *15*N.* A single-hop xGMI
     connection has a weight of 15.
   - PCIe segments are summed over all segments (GPU→CPU + CPU→CPU + CPU→GPU).
     Each GPU-to-CPU segment typically contributes 20. The CPU-to-CPU segment
     uses the actual io_link weight when available; if that weight cannot be
     read, a fallback value of 10 is used for that segment.

   Args:
       processor_handle_src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the source processor handle

       processor_handle_dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the destination processor handle

       weight (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an uint64_t to which the
           weight for the connection should be written.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_topo_get_link_weight(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst, uint64_t * weight)


.. py:function:: amdsmi_get_minmax_bandwidth_between_processors(processor_handle_src, processor_handle_dst, min_bandwidth, max_bandwidth)

   Retrieve minimal and maximal io link bandwidth between 2 GPUs

   @platform{gpu_bm_linux}

   Given a source processor handle ``processor_handle_src`` and
   a destination processor handle ``processor_handle_dst,``  pointer to an
   uint64_t ``min_bandwidth,`` and a pointer to uint64_t ``max_bandiwidth,``
   this function will write theoretical minimal and maximal bandwidth limits.
   API works if src and dst are connected via xgmi and have 1 hop distance.

   Args:
       processor_handle_src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the source processor handle

       processor_handle_dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the destination processor handle

       min_bandwidth (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an uint64_t to which the
           minimal bandwidth for the connection should be written.

       max_bandwidth (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an uint64_t to which the
           maximal bandwidth for the connection should be written.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst, uint64_t * min_bandwidth, uint64_t * max_bandwidth)


.. py:function:: amdsmi_topo_get_link_type(processor_handle_src, processor_handle_dst, hops, type)

   Retrieve the hops and the connection type between 2 GPUs

   @platform{gpu_bm_linux}

   Given a source processor handle ``processor_handle_src`` and
   a destination processor handle ``processor_handle_dst,`` and a pointer to an
   uint64_t ``hops`` and a pointer to an ::amdsmi_link_type_t ``type,``
   this function will write the number of hops and the connection type
   between the device ``processor_handle_src`` and ``processor_handle_dst`` to the memory
   pointed to by ``hops`` and ``type.``

   Note:
       The value written to ``hops`` is an **abstracted topology step count**,
       not the number of physical xGMI links traversed. The possible values are:

   | Value | Meaning |
   |-------|---------|
   | 1 | The two GPUs are reachable over xGMI, regardless of how many physical xGMI links the route traverses. |
   | 2 | The two GPUs communicate over PCIe within the same CPU NUMA node. |
   | 3 | The two GPUs communicate over PCIe across different CPU NUMA nodes. |
   | 4 | Fallback value used when the inter-CPU io_link weight cannot be read. |

   Two GPUs on the same xGMI fabric always report a hop count of 1, even when
   the data physically crosses several xGMI links. To obtain the literal number
   of physical xGMI links between two devices, read the value exposed by the
   ``amdgpu`` driver at ```/sys/class/drm/card{0,1,``…}/device/xgmi_num_hops` instead.

   Args:
       processor_handle_src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the source processor handle

       processor_handle_dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the destination processor handle

       hops (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an uint64_t to which the
           abstracted hop count for the connection should be written.

       type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an ::amdsmi_link_type_t to which the
           type for the connection should be written.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_topo_get_link_type(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst, uint64_t * hops, amdsmi_link_type_t * type)


.. py:function:: amdsmi_get_link_topology_nearest(processor_handle, link_type, topology_nearest_info)

   Retrieve the set of GPUs that are nearest to a given device
           at a specific interconnectivity level.

   @platform{gpu_bm_linux} @platform{host}

   Once called topology_nearest_info will get populated with a list of
   all nearest devices for a given link_type. The list has a count of
   the number of devices found and their respective handles/identifiers.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The identifier of the given device.

       link_type (:py:obj:`~.amdsmi_link_type_t`) -- *IN*:
           The amdsmi_link_type_t level to search for nearest GPUs.

       topology_nearest_info (:py:obj:`~.amdsmi_topology_nearest_t`/:py:obj:`~.object`) -- *IN,OUT*:
           .count;
                             - When zero, set to the number of matching GPUs such that .device_list can be
           malloc'd.
                             - When non-zero, .device_list will be filled with count number of
           processor_handle. .device_list An array of processor_handle for GPUs found at level.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_link_topology_nearest(amdsmi_processor_handle processor_handle, amdsmi_link_type_t link_type, amdsmi_topology_nearest_t * topology_nearest_info)


.. py:function:: amdsmi_is_P2P_accessible(processor_handle_src, processor_handle_dst, accessible)

   Return P2P availability status between 2 GPUs

   @platform{gpu_bm_linux}

   Given a source processor handle ``processor_handle_src`` and
   a destination processor handle ``processor_handle_dst,`` and a pointer to a
   bool ``accessible,`` this function will write the P2P connection status
   between the device ``processor_handle_src`` and ``processor_handle_dst`` to the memory
   pointed to by ``accessible.``

   Args:
       processor_handle_src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the source processor handle

       processor_handle_dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the destination processor handle

       accessible (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a bool to which the status for
           the P2P connection availability should be written.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_is_P2P_accessible(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst, _Bool * accessible)


.. py:function:: amdsmi_topo_get_p2p_status(processor_handle_src, processor_handle_dst, type, cap)

   Retrieve connection type and P2P capabilities between 2 GPUs

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}

   Given a source processor handle ``processor_handle_src`` and
    a destination processor handle ``processor_handle_dst,`` a pointer to an amdsmi_link_type_t `type`, and a pointer to amdsmi_p2p_capability_t ``cap.`` This function will write the connection
   type, and io link capabilities between the device
    ``processor_handle_src`` and ``processor_handle_dst`` to the memory
    pointed to by ``cap`` and ``type.``

   Args:
       processor_handle_src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the source processor handle

       processor_handle_dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           the destination processor handle

       type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an ::amdsmi_link_type_t to which the
           type for the connection should be written.

       cap (:py:obj:`~.amdsmi_p2p_capability_t`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to an ::amdsmi_p2p_capability_t to which the
           io link capabilities should be written.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_topo_get_p2p_status(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst, amdsmi_link_type_t * type, amdsmi_p2p_capability_t * cap)


.. py:function:: amdsmi_get_gpu_compute_partition(processor_handle, compute_partition, len)

   Retrieves the current compute partitioning for a desired device

   Deprecated:
       This API is slated for removal in a future ROCm release;
       ::amdsmi_get_gpu_accelerator_partition_profile() should be used instead

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a string ``compute_partition,``
   and uint32 ``len,`` this function will attempt to obtain the device's
   current compute partition setting string. Upon successful retrieval,
   the obtained device's compute partition settings string shall be stored in
   the passed ``compute_partition`` char string variable.

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
   @retval ::AMDSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function
   @retval ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if ``len`` bytes is not
   large enough to hold the entire compute partition value. In this case,
   only ``len`` bytes will be written.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       compute_partition (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *INOUT*:
           a pointer to a char string variable,
           which the device's current compute partition will be written to.

       len (:py:obj:`~.int`) -- *IN*:
           the length of the caller provided buffer ``compute_partition,``
           suggested length is 4 or greater.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_compute_partition(amdsmi_processor_handle processor_handle, char * compute_partition, uint32_t len)


.. py:function:: amdsmi_set_gpu_compute_partition(processor_handle, compute_partition)

   Modifies a selected device's compute partition setting.

   Deprecated:
       This API is slated for removal in a future ROCm release;
       ::amdsmi_set_gpu_accelerator_partition_profile() should be used instead

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle,`` a type of compute partition
   ``compute_partition,`` this function will attempt to update the selected
   device's compute partition setting. This function does not allow any concurrent operations.
   Device must be idle and have no workloads when performing set partition operations.

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_NO_PERM function requires admin/sudo privileges
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
   @retval ::AMDSMI_STATUS_SETTING_UNAVAILABLE the provided setting is
   unavailable for current device
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       compute_partition (:py:obj:`~.amdsmi_compute_partition_type_t`) -- *IN*:
           using enum ::amdsmi_compute_partition_type_t,
           define what the selected device's compute partition setting should be
           updated to.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_compute_partition(amdsmi_processor_handle processor_handle, amdsmi_compute_partition_type_t compute_partition)


.. py:function:: amdsmi_get_gpu_compute_partition_mem_alloc_mode(processor_handle)

   Retrieves the current compute partition memory allocation mode
   for a desired device.

   Deprecated:
       This API is slated for removal in a future ROCm release;
       ::amdsmi_get_gpu_accelerator_partition_mem_alloc_mode() should be used instead

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer
   ``mode,`` this function will attempt to obtain the device's current
   compute partition memory allocation mode. The mode controls how HBM
   capacity is distributed across XCPs within each memory partition:
   - ::AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING — each XCP is capped
     to an even share.
   - ::AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_ALL — each XCP may use the
     full memory partition size (useful when only one XCP is active).

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
   @retval ::AMDSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
   @retval ::AMDSMI_STATUS_FILE_ERROR problem accessing the sysfs file
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: (undocumented)
       * :py:obj:`~.amdsmi_compute_partition_mem_alloc_mode_t`:
               a pointer to an ::amdsmi_compute_partition_mem_alloc_mode_t
               variable, into which the device's current memory allocation mode will
               be written.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_compute_partition_mem_alloc_mode(amdsmi_processor_handle processor_handle, amdsmi_compute_partition_mem_alloc_mode_t * mode)


.. py:function:: amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(processor_handle)

   Retrieves the current accelerator partition memory allocation mode
   for a desired device.

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a pointer
   ``mode,`` this function will attempt to obtain the device's current
   accelerator partition memory allocation mode. The mode controls how HBM
   capacity is distributed across XCPs within each memory partition:
   - ::AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING — each XCP is capped
     to an even share.
   - ::AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_ALL — each XCP may use the
     full memory partition size (useful when only one XCP is active).

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
   @retval ::AMDSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
   @retval ::AMDSMI_STATUS_FILE_ERROR problem accessing the sysfs file
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: (undocumented)
       * :py:obj:`~.amdsmi_accelerator_partition_mem_alloc_mode_t`:
               a pointer to an ::amdsmi_accelerator_partition_mem_alloc_mode_t
               variable, into which the device's current memory allocation mode will
               be written.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(amdsmi_processor_handle processor_handle, amdsmi_accelerator_partition_mem_alloc_mode_t * mode)


.. py:function:: amdsmi_set_gpu_compute_partition_mem_alloc_mode(processor_handle, mode)

   Modifies a selected device's compute partition memory allocation mode.

   Deprecated:
       This API is slated for removal in a future ROCm release;
       ::amdsmi_set_gpu_accelerator_partition_mem_alloc_mode() should be used instead

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a mode
   ``mode,`` this function will attempt to update the selected device's
   compute partition memory allocation mode. The mode controls how HBM
   capacity is distributed across XCPs within each memory partition:
   - ::AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_CAPPING — each XCP is capped
     to an even share. This is the default.
   - ::AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_ALL — each XCP may use the
     full memory partition size.

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_NO_PERM function requires admin/sudo privileges
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
   @retval ::AMDSMI_STATUS_FILE_ERROR problem accessing the sysfs file
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to modify

       mode (:py:obj:`~.amdsmi_compute_partition_mem_alloc_mode_t`) -- *IN*:
           using enum ::amdsmi_compute_partition_mem_alloc_mode_t,
           define what the selected device's memory allocation mode should be
           updated to.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_compute_partition_mem_alloc_mode(amdsmi_processor_handle processor_handle, amdsmi_compute_partition_mem_alloc_mode_t mode)


.. py:function:: amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(processor_handle, mode)

   Modifies a selected device's compute partition memory allocation mode.

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a mode
   ``mode,`` this function will attempt to update the selected device's
   compute partition memory allocation mode. The mode controls how HBM
   capacity is distributed across XCPs within each memory partition:
   - ::AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING — each XCP is capped
     to an even share. This is the default.
   - ::AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_ALL — each XCP may use the
     full memory partition size.

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_NO_PERM function requires admin/sudo privileges
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
   @retval ::AMDSMI_STATUS_FILE_ERROR problem accessing the sysfs file
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to modify

       mode (:py:obj:`~.amdsmi_accelerator_partition_mem_alloc_mode_t`) -- *IN*:
           using enum ::amdsmi_accelerator_partition_mem_alloc_mode_t,
           define what the selected device's memory allocation mode should be
           updated to.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(amdsmi_processor_handle processor_handle, amdsmi_accelerator_partition_mem_alloc_mode_t mode)


.. py:function:: amdsmi_get_gpu_memory_partition(processor_handle, memory_partition, len)

   Retrieves the current memory partition for a desired device

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a string ``memory_partition,``
   and uint32 ``len,`` this function will attempt to obtain the device's
   memory partition string. Upon successful retrieval, the obtained device's
   memory partition string shall be stored in the passed ``memory_partition``
   char string variable.

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
   @retval ::AMDSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function
   @retval ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if ``len`` bytes is not
   large enough to hold the entire memory partition value. In this case,
   only ``len`` bytes will be written.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       memory_partition (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *INOUT*:
           a pointer to a char string variable,
           which the device's memory partition will be written to.

       len (:py:obj:`~.int`) -- *IN*:
           the length of the caller provided buffer ``memory_partition,``
           suggested length is 5 or greater.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_memory_partition(amdsmi_processor_handle processor_handle, char * memory_partition, uint32_t len)


.. py:function:: amdsmi_set_gpu_memory_partition(processor_handle, memory_partition)

   Modifies a selected device's current memory partition setting.

   Deprecated:
       This API is slated for removal in a future ROCm release;
       ::amdsmi_set_gpu_memory_partition_mode() should be used instead

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a type of memory partition
   ``memory_partition,`` this function will attempt to update the selected
   device's memory partition setting. This function does not allow any concurrent operations.
   Device must be idle and have no workloads when performing set partition operations.

   On @platform{gpu_bm_linux} AMDGPU driver restart is REQUIRED to complete updating to
   the new memory partition setting.

   @retval ::AMDSMI_STATUS_SUCCESS call was successful
   @retval ::AMDSMI_STATUS_NO_PERM function requires admin/sudo privileges
   @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
   @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
   support this function
   @retval ::AMDSMI_STATUS_BUSY device is busy, a resource or mutex could not be acquired

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       memory_partition (:py:obj:`~.amdsmi_memory_partition_type_t`) -- *IN*:
           using enum ::amdsmi_memory_partition_type_t,
           define what the selected device's current mode setting should be updated to.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_memory_partition(amdsmi_processor_handle processor_handle, amdsmi_memory_partition_type_t memory_partition)


.. py:function:: amdsmi_get_gpu_memory_partition_config(processor_handle)

   Returns current gpu memory partition capabilities

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_memory_partition_config_t`:
               reference to the memory partition config.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_memory_partition_config(amdsmi_processor_handle processor_handle, amdsmi_memory_partition_config_t * config)


.. py:function:: amdsmi_set_gpu_memory_partition_mode(processor_handle, mode)

   Sets memory partition mode
   Set memory partition setting based on memory_partition mode
   from amdsmi_get_gpu_memory_partition_config

   @platform{gpu_bm_linux} @platform{host}

   Given a processor handle ``processor_handle`` and a type of memory partition
   ``mode,`` this function will attempt to update the selected
   device's memory partition setting. This function does not allow any concurrent operations.
   Device must be idle and have no workloads when performing set partition operations.

   On @platform{gpu_bm_linux} AMDGPU driver restart is REQUIRED to complete updating
   to the new memory partition setting.

   On @platform{gpu_bm_linux} AMDGPU driver restart is REQUIRED to complete updating to
   the new memory partition setting.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           A processor handle

       mode (:py:obj:`~.amdsmi_memory_partition_type_t`) -- *IN*:
           Enum representing memory partitioning mode to set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_memory_partition_mode(amdsmi_processor_handle processor_handle, amdsmi_memory_partition_type_t mode)


.. py:function:: amdsmi_get_gpu_accelerator_partition_profile_config(processor_handle)

   Returns gpu accelerator partition caps as currently configured in the system

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}

   Note:
       API requires admin/sudo privileges or API will not be able to read all resources
       for @platform{gpu_bm_linux} or any resources for @platform{host}.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_accelerator_partition_profile_config_t`:
               reference to the accelerator partition config.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile_config(amdsmi_processor_handle processor_handle, amdsmi_accelerator_partition_profile_config_t * profile_config)


.. py:function:: amdsmi_get_gpu_accelerator_partition_profile(processor_handle, partition_id)

   Returns current gpu accelerator partition cap

   Note:
       API requires admin/sudo privileges or API will not be able to read all resources
       for @platform{gpu_bm_linux} or any resources for @platform{host}.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       partition_id (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           array of ids for current accelerator profile.
           Must be allocated by user.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_accelerator_partition_profile_t`:
               reference to the accelerator partition profile.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile(amdsmi_processor_handle processor_handle, amdsmi_accelerator_partition_profile_t * profile, uint32_t * partition_id)


.. py:function:: amdsmi_set_gpu_accelerator_partition_profile(processor_handle, profile_index)

   Set accelerator partition setting based on profile_index
   from amdsmi_get_gpu_accelerator_partition_profile_config

   @platform{gpu_bm_linux} @platform{host}

   Note:
       API requires admin/sudo privileges or API will not be able to read all resources
       for @platform{gpu_bm_linux} or any resources for @platform{host}.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       profile_index (:py:obj:`~.int`) -- *IN*:
           Represents index of a partition user wants to set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_accelerator_partition_profile(amdsmi_processor_handle processor_handle, uint32_t profile_index)


.. py:function:: amdsmi_init_gpu_event_notification(processor_handle)

   Prepare to collect event notifications for a GPU

   @platform{gpu_bm_linux}

   This function prepares to collect events for the GPU with device
   ID ``processor_handle,`` by initializing any required system parameters. This call
   may open files which will remain open until ::amdsmi_stop_gpu_event_notification()
   is called.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle corresponding to the device on which to
           listen for events

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_init_gpu_event_notification(amdsmi_processor_handle processor_handle)


.. py:function:: amdsmi_set_gpu_event_notification_mask(processor_handle, mask)

   Specify which events to collect for a device

   @platform{gpu_bm_linux}

   Given a processor handle ``processor_handle`` and a ``mask`` consisting of
   elements of ::amdsmi_evt_notification_type_t OR'd together, this function
   will listen for the events specified in ``mask`` on the device
   corresponding to ``processor_handle.``

   Note:
       AMDSMI_STATUS_INIT_ERROR is returned if
       ::amdsmi_init_gpu_event_notification() has not been called before a call to this
       function

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle corresponding to the device on which to
           listen for events

       mask (:py:obj:`~.int`) -- *IN*:
           Bitmask generated by OR'ing 1 or more elements of
           ::amdsmi_evt_notification_type_t indicating which event types to listen for,
           where the amdsmi_evt_notification_type_t value indicates the bit field, with
           bit position starting from 1.
           For example, if the mask field is 0x0000000000000003, which means first bit,
           bit 1 (bit position start from 1) and bit 2 are set, which indicate interest
           in receiving AMDSMI_EVT_NOTIF_VMFAULT (which has a value of 1) and
           AMDSMI_EVT_NOTIF_THERMAL_THROTTLE event (which has a value of 2).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle processor_handle, uint64_t mask)


.. py:function:: amdsmi_get_gpu_event_notification(timeout_ms, num_elem, data)

   Collect event notifications, waiting a specified amount of time

   @platform{gpu_bm_linux}

   Given a time period ``timeout_ms`` in milliseconds and a caller-
   provided buffer of ::amdsmi_evt_notification_data_t's ``data`` with a length
   (in ::amdsmi_evt_notification_data_t's, also specified by the caller) in the
   memory location pointed to by ``num_elem,`` this function will collect
   ::amdsmi_evt_notification_type_t events for up to ``timeout_ms`` milliseconds,
   and write up to *``num_elem`` event items to ``data.`` Upon return ``num_elem``
   is updated with the number of events that were actually written. If events
   are already present when this function is called, it will write the events
   to the buffer then poll for new events if there is still caller-provided
   buffer available to write any new events that would be found.

   This function requires prior calls to ::amdsmi_init_gpu_event_notification() and
   :: amdsmi_set_gpu_event_notification_mask(). This function polls for the
   occurrence of the events on the respective devices that were previously
   specified by :: amdsmi_set_gpu_event_notification_mask().

   Args:
       timeout_ms (:py:obj:`~.int`) -- *IN*:
           number of milliseconds to wait for an event
           to occur

       num_elem (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           pointer to uint32_t, provided by the caller. On
           input, this value tells how many ::amdsmi_evt_notification_data_t elements
           are being provided by the caller with ``data.`` On output, the location
           pointed to by ``num_elem`` will contain the number of items written to
           the provided buffer.

       data (:py:obj:`~.amdsmi_evt_notification_data_t`/:py:obj:`~.object`) -- *OUT*:
           pointer to a caller-provided memory buffer of size
           ``num_elem`` ::amdsmi_evt_notification_data_t to which this function may safely
           write. If there are events found, up to ``num_elem`` event items will be
           written to ``data.``

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_event_notification(int timeout_ms, uint32_t * num_elem, amdsmi_evt_notification_data_t * data)


.. py:function:: amdsmi_stop_gpu_event_notification(processor_handle)

   Close any file handles and free any resources used by event
   notification for a GPU

   @platform{gpu_bm_linux}

   Any resources used by event notification for the GPU with
   processor handle ``processor_handle`` will be free with this
   function. This includes freeing any memory and closing file handles. This
   should be called for every call to ::amdsmi_init_gpu_event_notification()

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The processor handle of the GPU for which event
           notification resources will be free

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_stop_gpu_event_notification(amdsmi_processor_handle processor_handle)


.. py:function:: amdsmi_get_gpu_driver_info(processor_handle)

   Returns the driver version information

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
   @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_driver_info_t`:
               Reference to driver information structure. Must be
               allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_driver_info(amdsmi_processor_handle processor_handle, amdsmi_driver_info_t * info)


.. py:function:: amdsmi_get_gpu_asic_info(processor_handle)

   Returns the ASIC information for the device

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
   @platform{guest_windows}

   This function returns ASIC information such as the product name,
   the vendor ID, the subvendor ID, the device ID,
   the revision ID and the serial number.

   Note:
       The processor_handle that contains amdsmi_asic_info_t member oam_id = 0
       corresponds to the socket that contains baseboard information.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_asic_info_t`:
               Reference to static asic information structure.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_asic_info(amdsmi_processor_handle processor_handle, amdsmi_asic_info_t * info)


.. py:function:: amdsmi_get_gpu_kfd_info(processor_handle)

   Returns the KFD (Kernel Fusion Driver) information for the device

   @platform{gpu_bm_linux}

   This function returns KFD information populated into the amdsmi_kfd_info_t.
   This contains the kfd_id and node_id which allow for the ID and
   index of this device in the KFD.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_kfd_info_t`:
               Reference to kfd information structure.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_kfd_info(amdsmi_processor_handle processor_handle, amdsmi_kfd_info_t * info)


.. py:function:: amdsmi_get_gpu_vram_info(processor_handle)

   Returns vram info

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           PF of a processor for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_vram_info_t`:
               Reference to vram info structure
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_vram_info(amdsmi_processor_handle processor_handle, amdsmi_vram_info_t * info)


.. py:function:: amdsmi_get_gpu_board_info(processor_handle)

   Returns the board part number and board information for the requested device

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_board_info_t`:
               Reference to board info structure.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_board_info(amdsmi_processor_handle processor_handle, amdsmi_board_info_t * info)


.. py:function:: amdsmi_get_power_cap_info(processor_handle, sensor_ind)

   Returns the power caps as currently configured in the system.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       sensor_ind (:py:obj:`~.int`) -- *IN*:
           A 0-based sensor index. Normally, this will be 0.
           If a device has more than one sensor, it could be greater than 0.
           Parameter ``sensor_ind`` is unused on @platform{host}.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_power_cap_info_t`:
               Reference to power caps information structure. Must be
               allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle, uint32_t sensor_ind, amdsmi_power_cap_info_t * info)


.. py:function:: amdsmi_get_pcie_info(processor_handle)

   Returns the PCIe info for the GPU.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_pcie_info_t`:
               Reference to the PCIe information
               returned by the library. Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_pcie_info(amdsmi_processor_handle processor_handle, amdsmi_pcie_info_t * info)


.. py:function:: amdsmi_get_gpu_xcd_counter(processor_handle, xcd_count)

   Returns the 'xcd_counter' from the GPU metrics associated with the device

   @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_mvf}

   @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
   ::AMDSMI_STATUS_NOT_SUPPORTED is returned in case the metric unit
     does not exist for the given device.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       xcd_count (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *INOUT*:
           a pointer to uint16_t to which the device gpu
           metric unit will be stored. Must be allocated by user.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_xcd_counter(amdsmi_processor_handle processor_handle, uint16_t * xcd_count)


.. py:function:: amdsmi_get_npm_info(node_handle)

   Retrieves node power management (NPM) status and power limit for the specified node.

   @platform{gpu_bm_linux} @platform{host}

   This function queries the NPM controller for the given node and returns whether NPM is
   enabled, along with the current node-level power limit in Watts. The NPM status and limit are set
   out-of-band and reported via this API.

   Args:
       node_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Handle to the Node to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: AMDSMI_STATUS_SUCCESS on success, non-zero on failure.
       * :py:obj:`~.amdsmi_npm_info_t`:
               Pointer to amdsmi_npm_info_t structure to receive NPM status and limit.
               Must be allocated by the user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_npm_info(amdsmi_node_handle node_handle, amdsmi_npm_info_t * info)


.. py:function:: amdsmi_get_fw_info(processor_handle)

   Returns the firmware versions running on the device.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
   @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_fw_info_t`:
               Reference to the fw info. Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_fw_info(amdsmi_processor_handle processor_handle, amdsmi_fw_info_t * info)


.. py:function:: amdsmi_get_gpu_vbios_info(processor_handle)

   Returns the static information for the vBIOS on the device.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
   @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_vbios_info_t`:
               Reference to static vBIOS information.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_vbios_info(amdsmi_processor_handle processor_handle, amdsmi_vbios_info_t * info)


.. py:function:: amdsmi_get_temp_metric(processor_handle, sensor_type, metric, temperature)

   Get the temperature metric value for the specified metric, from the
   specified temperature sensor on the specified device. It is not supported on
   virtual machine guest

   @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}

   Given a processor handle ``processor_handle,`` a sensor type ``sensor_type,`` a
   ::amdsmi_temperature_metric_t ``metric`` and a pointer to an int64_t `temperature`, this function will write the value of the metric indicated by
   ``metric`` and ``sensor_type`` to the memory location ``temperature.``

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           a processor handle

       sensor_type (:py:obj:`~.amdsmi_temperature_type_t`) -- *IN*:
           part of device from which temperature should be
           obtained. This should come from the enum ::amdsmi_temperature_type_t

       metric (:py:obj:`~.amdsmi_temperature_metric_t`) -- *IN*:
           enum indicated which temperature value should be
           retrieved

       temperature (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           a pointer to int64_t to which the temperature is in Celsius.
           If this parameter is nullptr, this function will return ::AMDSMI_STATUS_INVAL if the function
           is supported with the provided, arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not
           supported with the provided arguments.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_temp_metric(amdsmi_processor_handle processor_handle, amdsmi_temperature_type_t sensor_type, amdsmi_temperature_metric_t metric, int64_t * temperature)


.. py:function:: amdsmi_get_gpu_activity(processor_handle)

   Returns the current usage of the GPU engines (GFX, MM and MEM).
   Each usage is reported as a percentage from 0-100%.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_engine_usage_t`:
               Reference to the gpu engine usage structure. Must be allocated by user.
               When ``gfx_activity`` is unavailable it is reported as N/A using the sentinel
               0x0000FFFF (a uint16_t max value carried in the uint32_t field), not 0xFFFFFFFF.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_activity(amdsmi_processor_handle processor_handle, amdsmi_engine_usage_t * info)


.. py:function:: amdsmi_get_power_info(processor_handle)

   Returns the current power and voltage of the GPU.

   @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}

   Note:
       amdsmi_power_info_t::socket_power metric can rarely spike above the socket power limit in
       some cases

   Note:
       unsupported struct members are set to UINT32_MAX

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           PF of a processor for which  to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_power_info_t`:
               Reference to the gpu power structure. Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_power_info(amdsmi_processor_handle processor_handle, amdsmi_power_info_t * info)


.. py:function:: amdsmi_is_gpu_power_management_enabled(processor_handle)

   Returns is power management enabled

   @platform{gpu_bm_linux} @platform{host}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           PF of a processor for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.bool`:
               Reference to bool. Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_is_gpu_power_management_enabled(amdsmi_processor_handle processor_handle, _Bool * enabled)


.. py:function:: amdsmi_get_clock_info(processor_handle, clk_type)

   Returns the measurements of the clocks in the GPU
          for the GFX and multimedia engines and Memory. This call
          reports the averages over 1s in MHz. It is not supported
          on virtual machine guest

   @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       clk_type (:py:obj:`~.amdsmi_clk_type_t`) -- *IN*:
           Enum representing the clock type to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_clk_info_t`:
               Reference to the gpu clock structure.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_clock_info(amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_info_t * info)


.. py:function:: amdsmi_get_gpu_vram_usage(processor_handle)

   Returns the VRAM usage (both total and used memory) in MegaBytes.

   @platform{gpu_bm_linux} @platform{guest_windows}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_vram_usage_t`:
               Reference to vram information. Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_vram_usage(amdsmi_processor_handle processor_handle, amdsmi_vram_usage_t * info)


.. py:function:: amdsmi_get_violation_status(processor_handle)

   Returns the violations for a processor

   Warning: API will be slow due to polling driver for 2 samples. Require
   a minimum wait of 100ms between the 2 samples in order to calculate. Otherwise
   users would need to use amdsmi_get_gpu_metrics_info for BM. See that API's struct
   for calculations.

   @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_violation_status_t`:
               Reference to all violation status details available.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_violation_status(amdsmi_processor_handle processor_handle, amdsmi_violation_status_t * info)


.. py:function:: amdsmi_get_gpu_process_list(processor_handle, max_processes, list)

   Returns the list of process information running on a given GPU.
    If pdh.dll is not present on the system, this API returns
    AMDSMI_STATUS_NOT_SUPPORTED. Sum of the process memory is not expected to be the total memory
   usage.

   @platform{gpu_bm_linux} @platform{guest_windows}

   Warning:
       IMPORTANT: To get valid return values, at least 1 second needs to pass
       from starting the program to the first call of this function,
       and before every following call of this function after that, to get correct values

   Note:
       The user provides a buffer to store the list and the maximum
       number of processes that can be returned. If the user sets
       max_processes to 0, the current total number of processes will
       replace max_processes param. After that, the function needs to be
       called again, with updated max_processes, to successfully fill the
       process list, which was previously allocated with max_processes

   Note:
       If the reserved size for processes is smaller than the number of
       actual processes running. The AMDSMI_STATUS_OUT_OF_RESOURCES is
       an indication the caller should handle the situation (resize).
       The max_processes is always changed to reflect the actual size of
       list of processes running, so the caller knows where it is at.

   For cases where max_process is not zero (0), it specifies the list's size limit.
                    That is, the maximum size this list will be able to hold. After the list is
   built internally, as a return status, we will have AMDSMI_STATUS_OUT_OF_RESOURCES when the
   original size limit is smaller than the actual list of processes running. Hence, the caller is
   aware the list size needs to be resized, or AMDSMI_STATUS_SUCCESS otherwise. Holding a copy of
   max_process before it is passed in will be helpful for monitoring the allocations done upon each
   call since the max_process will permanently be changed to reflect the actual number of processes
   running.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

       max_processes (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           Reference to the size of the list buffer in
           number of elements. Returns the return number of elements
           in list or the number of running processes if equal to 0,
           and if given value in param max_processes is less than
           number of processes currently running,
           AMDSMI_STATUS_OUT_OF_RESOURCES will be returned.

       list (:py:obj:`~.amdsmi_proc_info_t`/:py:obj:`~.object`) -- *OUT*:
           Reference to a user-provided buffer where the process
           list will be returned. This buffer must contain at least
           max_processes entries of type amd_proc_info_list_t. Must be allocated
           by user.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success,
                                      | ::AMDSMI_STATUS_OUT_OF_RESOURCES, filled list buffer with data, but
           number of actual running processes is larger than the size provided.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_process_list(amdsmi_processor_handle processor_handle, uint32_t * max_processes, amdsmi_proc_info_t * list)


.. py:function:: amdsmi_get_gpu_process_list_by_pid(processor_handles, num_processors, max_processes)

   Get the list of processes running on one or more GPUs, grouped by PID.

   Aggregates per-GPU process lists across all provided processor handles
   and returns one entry per unique PID. Each entry contains the per-GPU breakdown
   for every GPU that PID is active on. Results are sorted ascending by PID.

   @platform{gpu_bm_linux}

   Args:
       processor_handles (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of processor handles to query

       num_processors (:py:obj:`~.int`) -- *IN*:
           Number of handles in processor_handles

       max_processes (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           On input: capacity of procs. On output: number of
           unique PIDs written (or required if procs is NULL).

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success,
                                      | ::AMDSMI_STATUS_OUT_OF_RESOURCES if max_processes was too small,
                                      | ::AMDSMI_STATUS_INVAL if processor_handles is NULL or num_processors
           is 0
       * :py:obj:`~.amdsmi_proc_info_by_pid_t`:
               Caller-allocated buffer of amdsmi_proc_info_by_pid_t.
               Pass NULL to query the required size via max_processes.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_process_list_by_pid(amdsmi_processor_handle * processor_handles, uint32_t num_processors, amdsmi_proc_info_by_pid_t * procs, uint32_t * max_processes)


.. py:function:: amdsmi_get_gpu_ptl_state(processor_handle)

   Get PTL enable/disable state

   @platform{gpu_bm_linux} @platform{host}

   This function retrieves whether PTL (Peak Tops Limiter) is currently
   enabled or disabled for the specified processor. This is a simple state query
   that returns the current PTL operational state without detailed configuration.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success,
           ::AMDSMI_STATUS_NOT_SUPPORTED if PTL is not supported on this device,
           non-zero on other failures
       * :py:obj:`~.bool`:
               Pointer to boolean that will be set to true if PTL is
               enabled, false if PTL is disabled

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_ptl_state(amdsmi_processor_handle processor_handle, _Bool * enabled)


.. py:function:: amdsmi_set_gpu_ptl_state(processor_handle, enable)

   Set PTL enable/disable state

   @platform{gpu_bm_linux} @platform{host}

   This function enables or disables PTL (Peak Tops Limiter) operation.
   Use amdsmi_set_gpu_ptl_enable_with_formats()
   for more control over the preferred data formats when enabling.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device to configure

       enable (:py:obj:`~.bint`) -- *IN*:
           Boolean flag: true to enable PTL with default formats,
           false to disable PTL

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_ptl_state(amdsmi_processor_handle processor_handle, _Bool enable)


.. py:function:: amdsmi_get_gpu_ptl_formats(processor_handle)

   Get PTL (Peak Tops Limiter) formats for the processor

   @platform{gpu_bm_linux} @platform{host}

   This function retrieves the current PTL formats
   for the specified processor. PTL prevents the product to never deliver more
   than a specified TOPS/second. If function returns 0 for both formats,
   PTL was never enabled before on that system

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device which to query

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success,
           ::AMDSMI_STATUS_NOT_SUPPORTED if PTL is not supported on this device,
           non-zero on other failures
       * :py:obj:`~.amdsmi_ptl_data_format_t`:
               Pointer to first preferred data format that receives peak performance
       * :py:obj:`~.amdsmi_ptl_data_format_t`:
               Pointer to second preferred data format that receives peak performance

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_ptl_formats(amdsmi_processor_handle processor_handle, amdsmi_ptl_data_format_t * data_format1, amdsmi_ptl_data_format_t * data_format2)


.. py:function:: amdsmi_set_gpu_ptl_formats(processor_handle, data_format1, data_format2)

   Set PTL with specified preferred data formats

   @platform{gpu_bm_linux} @platform{host}

   This function sets PTL with the specified preferred data format pair.
   PTL must be enabled first before calling this function using amdsmi_set_gpu_ptl_state.
   The two specified formats will receive accurate performance monitoring and peak
   performance. F8 and XF32 formats always receive peak performance regardless of this setting.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Device to configure

       data_format1 (:py:obj:`~.amdsmi_ptl_data_format_t`) -- *IN*:
           First preferred data format (must be from the limited set:
           I8, F16, BF16, F32, F64)

       data_format2 (:py:obj:`~.amdsmi_ptl_data_format_t`) -- *IN*:
           Second preferred data format (must be from the limited set:
           I8, F16, BF16, F32, F64, and different from data_format1)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success,
           ::AMDSMI_STATUS_NOT_SUPPORTED if PTL is not supported on this device,
           non-zero on other failures

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_ptl_formats(amdsmi_processor_handle processor_handle, amdsmi_ptl_data_format_t data_format1, amdsmi_ptl_data_format_t data_format2)


.. py:function:: amdsmi_get_cpu_handles(cpu_count, processor_handles)

   Get the list of cpu handles in the system.

   @platform{cpu_bm}

   Depends on AMDSMI_INIT_AMD_CPUS flag passed to ::amdsmi_init.
   The processor handles can be used in other APIs to get processor detail information.

   Args:
       cpu_count (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           As input, the value passed
           through this parameter is the number of ::amdsmi_processor_handle that
           may be safely written to the memory pointed to by ``processor_handles.`` This is the
           limit on how many processor handles will be written to ``processor_handles.`` On return, `socket_count` will contain the number of processor handles written to ``processor_handles,``
           or the number of processor handles that could have been written if enough memory had been
           provided.
           If ``processor_handles`` is NULL, as output, ``cpu_count`` will contain
           how many processors are available to read in the system.

       processor_handles (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a block of memory to which the
           ::amdsmi_processor_handle values will be written. This value may be NULL.
           In this case, this function can be used to query how many processors are
           available to read in the system.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_handles(uint32_t * cpu_count, amdsmi_processor_handle * processor_handles)


.. py:function:: amdsmi_get_cpucore_handles(cores_count, processor_handles)

   Get the list of the cpu core handles in a system.

   @platform{cpu_bm}

   This function retrieves the cpu core handles of a system.

   Args:
       cores_count (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           As input, the value passed
           through this parameter is the number of ::amdsmi_processor_handle's that
           may be safely written to the memory pointed to by ``processor_handles.`` This is the
           limit on how many core handles will be written to ``processor_handles.`` On return, `cores_count` will contain the number of core processor handles written to ``processor_handles,``
           or the number of core processor handles that could have been written if enough memory had been
           provided.
           If ``processor_handles`` is NULL, as output, ``processor_count`` will contain
           how many cpu cores are available to read in the system.

       processor_handles (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN,OUT*:
           A pointer to a block of memory to which the
           ::amdsmi_processor_handle values will be written. This value may be NULL.
           In this case, this function can be used to query how many processors are
           available to read.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpucore_handles(uint32_t * cores_count, amdsmi_processor_handle * processor_handles)


.. py:function:: amdsmi_get_cpu_core_energy(processor_handle, penergy)

   Get the core energy for a given core.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu core which to query

       penergy (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the core energy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_core_energy(amdsmi_processor_handle processor_handle, uint64_t * penergy)


.. py:function:: amdsmi_get_cpu_socket_energy(processor_handle, penergy)

   Get the socket energy for a given socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       penergy (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the socket energy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_energy(amdsmi_processor_handle processor_handle, uint64_t * penergy)


.. py:function:: amdsmi_get_threads_per_core(threads_per_core)

   Get Number of threads Per Core.

   @platform{cpu_bm}

   Args:
       threads_per_core (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the Number of threads Per Core

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_threads_per_core(uint32_t * threads_per_core)


.. py:function:: amdsmi_get_cpu_hsmp_driver_version(processor_handle, amdsmi_hsmp_driver_ver)

   Get HSMP Driver Version.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       amdsmi_hsmp_driver_ver (:py:obj:`~.amdsmi_hsmp_driver_version_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the HSMP Driver version

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_hsmp_driver_version(amdsmi_processor_handle processor_handle, amdsmi_hsmp_driver_version_t * amdsmi_hsmp_driver_ver)


.. py:function:: amdsmi_get_cpu_smu_fw_version(processor_handle, amdsmi_smu_fw)

   Get SMU Firmware Version.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       amdsmi_smu_fw (:py:obj:`~.amdsmi_smu_fw_version_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the firmware version

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_smu_fw_version(amdsmi_processor_handle processor_handle, amdsmi_smu_fw_version_t * amdsmi_smu_fw)


.. py:function:: amdsmi_get_cpu_hsmp_proto_ver(processor_handle, proto_ver)

   Get HSMP protocol Version.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       proto_ver (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the protocol version

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_hsmp_proto_ver(amdsmi_processor_handle processor_handle, uint32_t * proto_ver)


.. py:function:: amdsmi_get_cpu_prochot_status(processor_handle, prochot)

   Get normalized status of the processor's PROCHOT status.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       prochot (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the procohot status.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_prochot_status(amdsmi_processor_handle processor_handle, uint32_t * prochot)


.. py:function:: amdsmi_get_cpu_fclk_mclk(processor_handle, fclk, mclk)

   Get Data fabric clock and Memory clock in MHz.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       fclk (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return fclk

       mclk (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return mclk

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_fclk_mclk(amdsmi_processor_handle processor_handle, uint32_t * fclk, uint32_t * mclk)


.. py:function:: amdsmi_get_cpu_cclk_limit(processor_handle, cclk)

   Get core clock in MHz.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       cclk (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return core clock

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_cclk_limit(amdsmi_processor_handle processor_handle, uint32_t * cclk)


.. py:function:: amdsmi_get_cpu_socket_current_active_freq_limit(processor_handle, freq, src_type)

   Get current active frequency limit of the socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       freq (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return frequency value in MHz

       src_type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return frequency source name

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_current_active_freq_limit(amdsmi_processor_handle processor_handle, uint16_t * freq, char ** src_type)


.. py:function:: amdsmi_get_cpu_socket_freq_range(processor_handle, fmax, fmin)

   Get socket frequency range.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       fmax (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return maximum frequency

       fmin (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return minimum frequency

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_freq_range(amdsmi_processor_handle processor_handle, uint16_t * fmax, uint16_t * fmin)


.. py:function:: amdsmi_get_cpu_core_current_freq_limit(processor_handle, freq)

   Get socket frequency limit of the core.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu core which to query

       freq (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return frequency.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_core_current_freq_limit(amdsmi_processor_handle processor_handle, uint32_t * freq)


.. py:function:: amdsmi_set_cpu_rail_isofreq_policy(processor_handle, rail_isofreq_policy)

   Set CPU rail isolated frequency policy for independent core clock control per power rail

   This API configures the frequency policy for CPU power rails.
    - If a socket-wide limit (e.g., PPT) is setting the core clock frequency, this setting has no
   effect.
    - For other limiters specific to CPU power rails (e.g., TDC),
      this policy enables or disables independent core clocks per rail (VDDCR_CPU0 or VDDCR_CPU1).

    Policy values:
    - 0: Disable independent control (all cores on both rails have the same frequency limit)
    - 1: Enable independent control (each rail has an independent frequency limit)

    @platform{cpu_bm}

   par

   t

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           socket which to query

           @

       rail_isofreq_policy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           put buffer to store policy value, indicating the CPU
                                                 rail ISO frequency Policy setting:
                                 - 0: Disable independent control - each rail has its own independent
                                      frequency limit.
                                 - 1: Enable independent control - all cores on both rails share the same
                                      frequency limit.
           @re

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: mdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_rail_isofreq_policy(amdsmi_processor_handle processor_handle, _Bool * rail_isofreq_policy)


.. py:function:: amdsmi_get_cpu_rail_isofreq_policy(processor_handle, rail_isofreq_policy)

   Get CPU rail isolated frequency policy status for independent core clock control per
   power rail.

   This API retrieves the current frequency policy configuration for CPU power rails.
    - If a socket-wide limit (e.g., PPT) is setting the core clock frequency, the effective policy
   may be overridden.
    - For other limiters specific to CPU power rails (e.g., TDC),
      this policy indicates whether independent core clocks per rail (VDDCR_CPU0 or VDDCR_CPU1) are
   enabled or disabled.

    Policy values returned:
    - 0: Independent control disabled (all cores on both rails have the same frequency limit)
    - 1: Independent control enabled (each rail has an independent frequency limit)

    @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       rail_isofreq_policy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to receive the current cpu rail isolated
           frequency policy

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success, non-zero on failure

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_rail_isofreq_policy(amdsmi_processor_handle processor_handle, uint8_t * rail_isofreq_policy)


.. py:function:: amdsmi_set_cpu_dfc_ctrl(processor_handle, dfc_ctrl)

   Set the DFCState enabling control.

   DFCState control values for setting:
   - 0: Disable DFC control
   - 1: Enable DFC control

   @platform{cpu_bm}

   par

   ret

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           socket handle to query

           @

       dfc_ctrl (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           put buffer indicating whether to enable (1) or disable (0) DFCState
           control

            @

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: mdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success, non-zero on failure

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_dfc_ctrl(amdsmi_processor_handle processor_handle, uint8_t * dfc_ctrl)


.. py:function:: amdsmi_get_cpu_dfc_ctrl(processor_handle, dfc_ctrl)

   Get the current DFCState enabling control status.

   Returned DFCState control values:
   - 0: DFC control is disabled
   - 1: DFC control is enabled

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket handle to query

       dfc_ctrl (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to receive the current DFCState control status

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success, non-zero on failure

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_dfc_ctrl(amdsmi_processor_handle processor_handle, uint8_t * dfc_ctrl)


.. py:function:: amdsmi_get_cpu_core_boostlimit(processor_handle, pboostlimit)

   Get the core boost limit.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu core which to query

       pboostlimit (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill the boostlimit value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_core_boostlimit(amdsmi_processor_handle processor_handle, uint32_t * pboostlimit)


.. py:function:: amdsmi_get_cpu_socket_c0_residency(processor_handle, pc0_residency)

   Get the socket c0 residency.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       pc0_residency (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill the c0 residency value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_c0_residency(amdsmi_processor_handle processor_handle, uint32_t * pc0_residency)


.. py:function:: amdsmi_set_cpu_core_boostlimit(processor_handle, boostlimit)

   Set the core boostlimit value.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu core which to query

       boostlimit (:py:obj:`~.int`) -- *IN*:
           - boostlimit value to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_core_boostlimit(amdsmi_processor_handle processor_handle, uint32_t boostlimit)


.. py:function:: amdsmi_set_cpu_socket_boostlimit(processor_handle, boostlimit)

   Set the socket boostlimit value.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       boostlimit (:py:obj:`~.int`) -- *IN*:
           - boostlimit value to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_socket_boostlimit(amdsmi_processor_handle processor_handle, uint32_t boostlimit)


.. py:function:: amdsmi_get_cpu_core_floor_freq_limit(processor_handle, floor_freq)

   Get the CPU core floor limit frequency.

   This function retrieves the floor frequency limit for the specified CPU core.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU core which to query

       floor_freq (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill the floor limit frequency in MHz

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_core_floor_freq_limit(amdsmi_processor_handle processor_handle, uint32_t * floor_freq)


.. py:function:: amdsmi_get_cpu_floor_freq_limit(processor_handle, floor_freq)

   Get the CPU floor limit frequency.

   This function retrieves the floor frequency limit for the specified CPU socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU socket which to query

       floor_freq (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill the floor limit frequency in MHz

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_floor_freq_limit(amdsmi_processor_handle processor_handle, uint32_t * floor_freq)


.. py:function:: amdsmi_get_cpu_core_eff_floor_freq_limit(processor_handle, eff_floor_freq)

   Get the CPU core effective floor limit frequency.

   This function returns the effective floor frequency limit for the specified CPU core.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU core which to query

       eff_floor_freq (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill the effective floor limit frequency in MHz

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_core_eff_floor_freq_limit(amdsmi_processor_handle processor_handle, uint32_t * eff_floor_freq)


.. py:function:: amdsmi_get_cpu_eff_floor_freq_limit(processor_handle, eff_floor_freq)

   Get the CPU effective floor limit frequency.

   This function returns the effective floor frequency limit for the specified CPU socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU socket which to query

       eff_floor_freq (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill the effective floor limit frequency in MHz

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_eff_floor_freq_limit(amdsmi_processor_handle processor_handle, uint32_t * eff_floor_freq)


.. py:function:: amdsmi_set_cpu_core_floor_freq_limit(processor_handle, floor_freq)

   Set the CPU core floor limit frequency.

   This function sets the floor frequency limit for the specified CPU core.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU core which to set

       floor_freq (:py:obj:`~.int`) -- *IN*:
           - floor limit frequency value to be set in MHz

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_core_floor_freq_limit(amdsmi_processor_handle processor_handle, uint32_t floor_freq)


.. py:function:: amdsmi_set_cpu_floor_freq_limit(processor_handle, floor_freq)

   Set the CPU socket floor limit frequency.

   This function sets the floor frequency limit for the specified CPU socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU socket which to set

       floor_freq (:py:obj:`~.int`) -- *IN*:
           - floor limit frequency value to be set in MHz

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_floor_freq_limit(amdsmi_processor_handle processor_handle, uint32_t floor_freq)


.. py:function:: amdsmi_set_cpu_msr_floor_freq_limit(processor_handle, msr_floor_freq)

   Set CPU floor limit frequency via MSR(Model Specific Register).

   This function sets the floor frequency limit via MSR(Model Specific Register) for the
   specified CPU socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU socket which to set

       msr_floor_freq (:py:obj:`~.int`) -- *IN*:
           - MSR floor limit frequency value to be set in MHz

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_msr_floor_freq_limit(amdsmi_processor_handle processor_handle, uint32_t msr_floor_freq)


.. py:function:: amdsmi_set_cpu_core_msr_floor_freq_limit(processor_handle, msr_floor_freq)

   Set CPU core MSR floor limit frequency.

   This function sets the MSR floor frequency limit for the specified CPU core.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU core which to set

       msr_floor_freq (:py:obj:`~.int`) -- *IN*:
           - MSR floor limit frequency value to be set in MHz

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_core_msr_floor_freq_limit(amdsmi_processor_handle processor_handle, uint32_t msr_floor_freq)


.. py:function:: amdsmi_get_cpu_freq_range()

   Get the CPU socket frequency range.

   This function retrieves frequency limit range for CPU socket 0.

   @platform{cpu_bm}

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               - Output buffer to retrieve maximum frequency in MHz
       * :py:obj:`~.int`:
               - Output buffer to retrieve minimum frequency in MHz

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_freq_range(uint32_t * fmax, uint32_t * fmin)


.. py:function:: amdsmi_set_cpu_sdps_limit(processor_handle, sdps_limit)

   Set the SDPS(Socket DIMM Power Sloshing) limit for a given processor socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Processor handle for which to set the limit

       sdps_limit (:py:obj:`~.int`) -- *IN*:
           - SDPS limit value in milliwatts

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_sdps_limit(amdsmi_processor_handle processor_handle, uint32_t sdps_limit)


.. py:function:: amdsmi_get_cpu_sdps_limit(processor_handle)

   Get the current SDPS limit for a given processor socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Processor handle for which to query the limit

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               - Input buffer to receive the current SDPS limit value in
               milliwatts (mW)

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_sdps_limit(amdsmi_processor_handle processor_handle, uint32_t * sdps_limit)


.. py:function:: amdsmi_get_cpu_ddr_bw(processor_handle, ddr_bw)

   Get the DDR bandwidth data.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       ddr_bw (:py:obj:`~.amdsmi_ddr_bw_metrics_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill ddr bandwidth data

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_ddr_bw(amdsmi_processor_handle processor_handle, amdsmi_ddr_bw_metrics_t * ddr_bw)


.. py:function:: amdsmi_get_cpu_socket_temperature(processor_handle, ptmon)

   Get socket temperature.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       ptmon (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill temperature value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_temperature(amdsmi_processor_handle processor_handle, uint32_t * ptmon)


.. py:function:: amdsmi_get_cpu_tdelta(processor_handle)

   Read Thermal Delta (TDELTA) Behavior

   This API retrieves the thermal solution behavior value from the CPU socket

   Thermal Behavior values returned:
   - 0: Thermal solution behavior is normal (operating within expected thermal range)
   - 1 or any other value: Thermal solution is out of expected range (thermal stress detected)

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU socket handle to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: ccessful register read, non-zero on failure
       * :py:obj:`~.int`:
               - input buffer to store the thermal delta behavior value:
                                               - 0: Normal thermal solution behavior
                                               - Non-zero: Thermal solution out of expected range
               @return ::amdsmi_status_t
                       ::AMDSMI_STATUS_SUCCES

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_tdelta(amdsmi_processor_handle processor_handle, uint8_t * tdelta)


.. py:function:: amdsmi_get_cpu_svi3_vr_controller_temp(processor_handle, rail_selection, rail_index)

   Get Temperature of SVI3 VR(Voltage Rail)

   This API retrieves the temperature of SVI3 voltage regulator

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU socket handle to query

       rail_selection (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to to store rail_selection, rail_selection:
           0=HottestRail, 1=IndividualRail

       rail_index (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to store rail_index, rail_index must be given when
           rail_selection = 1 rail_index: 0->VDDCR_CPU0,1->VDDCR_CPU1,2->VDDCR_SOC,3->VDDIO,4->VDDIO_MEM_S3

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on successful temperature read, non-zero on failure
       * :py:obj:`~.int`:
               - Output buffer to retrieve the temperature value

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_svi3_vr_controller_temp(amdsmi_processor_handle processor_handle, uint32_t * rail_selection, uint32_t * rail_index, uint32_t * temp)


.. py:function:: amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(processor_handle, dimm_addr, rate)

   Get DIMM temperature range and refresh rate.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       dimm_addr (:py:obj:`~.int`) -- *IN*:
           - DIMM address

       rate (:py:obj:`~.amdsmi_temp_range_refresh_rate_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill temperature range and refresh rate value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(amdsmi_processor_handle processor_handle, uint8_t dimm_addr, amdsmi_temp_range_refresh_rate_t * rate)


.. py:function:: amdsmi_get_cpu_dimm_power_consumption(processor_handle, dimm_addr, dimm_pow)

   Get DIMM power consumption.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       dimm_addr (:py:obj:`~.int`) -- *IN*:
           - DIMM address

       dimm_pow (:py:obj:`~.amdsmi_dimm_power_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill power consumption value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_dimm_power_consumption(amdsmi_processor_handle processor_handle, uint8_t dimm_addr, amdsmi_dimm_power_t * dimm_pow)


.. py:function:: amdsmi_get_cpu_dimm_thermal_sensor(processor_handle, dimm_addr, dimm_temp)

   Get DIMM thermal sensor value.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       dimm_addr (:py:obj:`~.int`) -- *IN*:
           - DIMM address

       dimm_temp (:py:obj:`~.amdsmi_dimm_thermal_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill temperature value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_dimm_thermal_sensor(amdsmi_processor_handle processor_handle, uint8_t dimm_addr, amdsmi_dimm_thermal_t * dimm_temp)


.. py:function:: amdsmi_get_cpu_dimm_sb_reg(processor_handle, dimm_addr, lid, reg_offset, reg_space, data)

   Read DIMM sideband register data

   @platform{cpu_bm}

   ad

   ES

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Processor handle for the target socket

       dimm_addr (:py:obj:`~.int`) -- *IN*:
           - DIMM address

       lid (:py:obj:`~.int`) -- *IN*:
           - The local identifier of the device on DIMM

       reg_offset (:py:obj:`~.int`) -- *IN*:
           - Register offset within the specified register space

       reg_space (:py:obj:`~.int`) -- *IN*:
           - Register space selector:
                                       - 0: Volatile register space
                                       - 1: Non-volatile memory (NVM) register space

           @param[out] data - Input buffer to store the 4-byte re

       data (:py:obj:`~.rocm.bindings.util.types.ListOfUnsigned`/:py:obj:`~.object`) -- *OUT*:
           gister

           @return ::amdsmi_status_t
                   ::AMDSMI_STATUS_SUCC

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: ccessful register read, non-zero on failure

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_dimm_sb_reg(amdsmi_processor_handle processor_handle, uint32_t dimm_addr, uint32_t lid, uint32_t reg_offset, uint32_t reg_space, uint32_t * data)


.. py:function:: amdsmi_set_cpu_dimm_sb_reg(processor_handle, dimm_addr, lid, reg_offset, reg_space, write_data)

   Write Data to DIMM Sideband Register

   @platform{cpu_bm}

   t

   ES

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Processor handle for the target socket

       dimm_addr (:py:obj:`~.int`) -- *IN*:
           - DIMM address

       lid (:py:obj:`~.int`) -- *IN*:
           - The local identifier of the device on DIMM

       reg_offset (:py:obj:`~.int`) -- *IN*:
           - Register offset within the specified register space

       reg_space (:py:obj:`~.int`) -- *IN*:
           - Register space selector:
                                       - 0: Volatile register space
                                       - 1: Non-volatile memory (NVM) register space

           @param[in]  write_data - 4-byte data value to write to

       write_data (:py:obj:`~.int`) -- *IN*:
           turn ::amdsmi_status_t
           ::AMDSMI_STATUS_SUCC

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: ccessful register write, non-zero on failure

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_dimm_sb_reg(amdsmi_processor_handle processor_handle, uint32_t dimm_addr, uint32_t lid, uint32_t reg_offset, uint32_t reg_space, uint32_t write_data)


.. py:function:: amdsmi_set_cpu_xgmi_width(processor_handle, min, max)

   Set xgmi width.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       min (:py:obj:`~.int`) -- *IN*:
           - Minimum xgmi width to be set

       max (:py:obj:`~.int`) -- *IN*:
           - maximum xgmi width to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_xgmi_width(amdsmi_processor_handle processor_handle, uint8_t min, uint8_t max)


.. py:function:: amdsmi_set_cpu_gmi3_link_width_range(processor_handle, min_link_width, max_link_width)

   Set gmi3 link width range.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       min_link_width (:py:obj:`~.int`) -- *IN*:
           - minimum link width to be set.

       max_link_width (:py:obj:`~.int`) -- *IN*:
           - maximum link width to be set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_gmi3_link_width_range(amdsmi_processor_handle processor_handle, uint8_t min_link_width, uint8_t max_link_width)


.. py:function:: amdsmi_cpu_apb_enable(processor_handle)

   Enable APB.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_cpu_apb_enable(amdsmi_processor_handle processor_handle)


.. py:function:: amdsmi_cpu_apb_disable(processor_handle, pstate)

   Disable APB.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       pstate (:py:obj:`~.int`) -- *IN*:
           - pstate value to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_cpu_apb_disable(amdsmi_processor_handle processor_handle, uint8_t pstate)


.. py:function:: amdsmi_set_cpu_socket_lclk_dpm_level(processor_handle, nbio_id, min, max)

   Set NBIO lclk dpm level value.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       nbio_id (:py:obj:`~.int`) -- *IN*:
           - nbio index

       min (:py:obj:`~.int`) -- *IN*:
           - minimum value to be set

       max (:py:obj:`~.int`) -- *IN*:
           - maximum value to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_socket_lclk_dpm_level(amdsmi_processor_handle processor_handle, uint8_t nbio_id, uint8_t min, uint8_t max)


.. py:function:: amdsmi_get_cpu_socket_lclk_dpm_level(processor_handle, nbio_id, nbio)

   Get NBIO LCLK dpm level.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       nbio_id (:py:obj:`~.int`) -- *IN*:
           - nbio index

       nbio (:py:obj:`~.amdsmi_dpm_level_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill lclk dpm level

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_lclk_dpm_level(amdsmi_processor_handle processor_handle, uint8_t nbio_id, amdsmi_dpm_level_t * nbio)


.. py:function:: amdsmi_set_cpu_pcie_link_rate(processor_handle, rate_ctrl, prev_mode)

   Set pcie link rate.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       rate_ctrl (:py:obj:`~.int`) -- *IN*:
           - rate control value to be set.

       prev_mode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill previous rate control value.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_pcie_link_rate(amdsmi_processor_handle processor_handle, uint8_t rate_ctrl, uint8_t * prev_mode)


.. py:function:: amdsmi_set_cpu_df_pstate_range(processor_handle, min_pstate, max_pstate)

   Set df pstate range.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       min_pstate (:py:obj:`~.int`) -- *IN*:
           - minimum pstate value to be set

       max_pstate (:py:obj:`~.int`) -- *IN*:
           - maximum pstate value to be set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_df_pstate_range(amdsmi_processor_handle processor_handle, uint8_t min_pstate, uint8_t max_pstate)


.. py:function:: amdsmi_set_cpu_xgmi_pstate_range(processor_handle, min_pstate, max_pstate)

   Set the Min and Max XGMI PState Range

   This API configures the XGMI P-State range for the specified processor socket.

   P-State range constraints:
   - min_pstate: Minimum allowed XGMI P-State
   - max_pstate: Maximum allowed XGMI P-State
   - Constraint: max_pstate <= min_pstate

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to configure

       min_pstate (:py:obj:`~.int`) -- *IN*:
           - minimum XGMI P-State value

       max_pstate (:py:obj:`~.int`) -- *IN*:
           - maximum XGMI P-State value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_xgmi_pstate_range(amdsmi_processor_handle processor_handle, uint8_t min_pstate, uint8_t max_pstate)


.. py:function:: amdsmi_get_cpu_xgmi_pstate_range(processor_handle)

   Get the Max and Min XGMI PState Range

   This API retrieves the current XGMI P-State range configuration for the specified processor
   socket.

    P-State range values returned:
    - min_pstate: Current minimum XGMI P-State setting
    - max_pstate: Current maximum XGMI P-State setting
    - Relationship: max_pstate <= min_pstate

    @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               - Input buffer to store current minimum XGMI P-State value
       * :py:obj:`~.int`:
               - Input buffer to store current maximum XGMI P-State value

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_xgmi_pstate_range(amdsmi_processor_handle processor_handle, uint8_t * min_pstate, uint8_t * max_pstate)


.. py:function:: amdsmi_get_cpu_pc6_enable(processor_handle)

   Get the PC6 Enable State

   This API retrieves the current PC6 (Package C-state 6) enable state for the specified processor
   socket.

    PC6 enable state values returned:
    - 0: PC6 disabled
    - 1: PC6 enabled

    @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.int`:
               - Input buffer to store the current PC6 enable state

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_pc6_enable(amdsmi_processor_handle processor_handle, uint8_t * enabled)


.. py:function:: amdsmi_set_cpu_pc6_enable(processor_handle, enable)

   Set the PC6 Enable State

   This API configures the PC6 (Package C-state 6) enable state for the specified processor socket.

   PC6 enable state values:
   - 0: PC6 disabled
   - 1: PC6 enabled

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to configure

       enable (:py:obj:`~.int`) -- *IN*:
           - PC6 enable state (0=disable, 1=enable)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_pc6_enable(amdsmi_processor_handle processor_handle, uint8_t enable)


.. py:function:: amdsmi_get_cpu_cc6_enable(processor_handle, enabled)

   Get the Core C6 Enable State.

   This API retrieves the Core C6 (CC6) low-power state for the specified processor socket.

   CC6 enable state values returned:
   - 0: CC6 state disabled
   - 1: CC6 state enabled

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       enabled (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to store the current CC6 enable state

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success, non-zero on failure

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_cc6_enable(amdsmi_processor_handle processor_handle, uint8_t * enabled)


.. py:function:: amdsmi_set_cpu_cc6_enable(processor_handle, enable)

   Set the Core C6 Enable State.

   This API configures the Core C6 (CC6) low-power state for the specified processor socket.

   CC6 enable state values:
   - 0: Disable CC6 state
   - 1: Enable CC6 state

   @platform{cpu_bm}

   TU

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to configure

       enable (:py:obj:`~.int`) -- *IN*:
           - CC6 enable state value:
                               - 0: Disable CC6 low-power state
                               - 1: Enable CC6 low-power state

           @return ::amdsmi_status_t | ::AMDSMI_STA

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: SS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_cpu_cc6_enable(amdsmi_processor_handle processor_handle, uint8_t enable)


.. py:function:: amdsmi_get_cpu_current_io_bandwidth(processor_handle, link, io_bw)

   Get current input output bandwidth.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       link (:py:obj:`~.amdsmi_link_id_bw_type_t`) -- *IN*:
           - link id and bw type to which io bandwidth to be obtained

       io_bw (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill bandwidth data

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_current_io_bandwidth(amdsmi_processor_handle processor_handle, amdsmi_link_id_bw_type_t link, uint32_t * io_bw)


.. py:function:: amdsmi_get_cpu_current_xgmi_bw(processor_handle, link, xgmi_bw)

   Get current input output bandwidth.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       link (:py:obj:`~.amdsmi_link_id_bw_type_t`) -- *IN*:
           - link id and bw type to which xgmi bandwidth to be obtained

       xgmi_bw (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill bandwidth data

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_current_xgmi_bw(amdsmi_processor_handle processor_handle, amdsmi_link_id_bw_type_t link, uint32_t * xgmi_bw)


.. py:function:: amdsmi_get_hsmp_metrics_table_version(processor_handle, metrics_version)

   Get HSMP metrics table version

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       metrics_version (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           input buffer to return the HSMP metrics table version.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_hsmp_metrics_table_version(amdsmi_processor_handle processor_handle, uint32_t * metrics_version)


.. py:function:: amdsmi_get_hsmp_metrics_table(processor_handle, metrics_table)

   Get HSMP metrics table

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       metrics_table (:py:obj:`~.amdsmi_hsmp_metrics_table_t`/:py:obj:`~.object`) -- *IN,OUT*:
           input buffer to return the HSMP metrics table.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_hsmp_metrics_table(amdsmi_processor_handle processor_handle, amdsmi_hsmp_metrics_table_t * metrics_table)


.. py:function:: amdsmi_first_online_core_on_cpu_socket(processor_handle, pcore_ind)

   Get first online core on socket.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

       pcore_ind (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to fill first online core on socket data

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_first_online_core_on_cpu_socket(amdsmi_processor_handle processor_handle, uint32_t * pcore_ind)


.. py:function:: amdsmi_get_cpu_family(cpu_family)

   Get CPU family.

   @platform{cpu_bm}

   Args:
       cpu_family (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the cpu family

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_family(uint32_t * cpu_family)


.. py:function:: amdsmi_get_cpu_model(cpu_model)

   Get CPU model.

   @platform{cpu_bm}

   Args:
       cpu_model (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the cpu model

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_model(uint32_t * cpu_model)


.. py:function:: amdsmi_get_cpu_model_name(processor_handle)

   Retrieve the CPU processor model name based on the processor index.

   @platform{cpu_bm}

   This function obtains the CPU model name associated with the specified processor index
   from the list of available processor handles. Before invoking this function, ensure that
   the list of processor handles is properly initialized and that the processor type is specified.
   This function is to be utilized for RDC and is not part of ESMI library.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Cpu socket which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t indicating the result of the operation.
           - ::AMDSMI_STATUS_SUCCESS on successful retrieval of the model name.
           - A non-zero error code if the operation fails.
       * :py:obj:`~.amdsmi_cpu_info_t`:
               A pointer to an `amdsmi_cpu_info_t` structure that will be populated with the
               CPU processor model information upon successful execution of the function.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_model_name(amdsmi_processor_handle processor_handle, amdsmi_cpu_info_t * cpu_info)


.. py:function:: amdsmi_get_esmi_err_msg(status, status_string)

   Get a description of provided AMDSMI error status for esmi errors.

   @platform{cpu_bm}

   Set the provided pointer to a const char *, ``status_string,`` to
   a string containing a description of the provided error code ``status.``

   Args:
       status (:py:obj:`~.amdsmi_status_t`) -- *IN*:
           - The error status for which a description is desired.

       status_string (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN,OUT*:
           - A pointer to a const char * which will be made
           to point to a description of the provided error code

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_esmi_err_msg(amdsmi_status_t status, const char ** status_string)


.. py:function:: amdsmi_get_cpu_cores_per_socket(sock_count, soc_info)

   Get cpu cores per socket from sys filesystem.

   @platform{cpu_bm}

   Args:
       sock_count (:py:obj:`~.int`) -- *IN*:
           - cpu socket count

       soc_info (:py:obj:`~.amdsmi_sock_info_t`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the cpu cores per socket

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_cores_per_socket(uint32_t sock_count, amdsmi_sock_info_t * soc_info)


.. py:function:: amdsmi_get_cpu_socket_count(sock_count)

   Get CPU socket count from sys filesystem.

   @platform{cpu_bm}

   Args:
       sock_count (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to return the cpu socket count

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_socket_count(uint32_t * sock_count)


.. py:function:: amdsmi_get_cpu_enabled_commands(processor_handle, r_mask, mask0, mask1, mask2)

   Get HSMP Enabled Commands information for a given CPU socket.

   This function retrieves enabled commands bit masks for both read and write commands
   from the HSMP interface.

   @platform{cpu_bm}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           CPU socket handle to query

       r_mask (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           - Input buffer to store read mask, indicating read or write enabled commands
           to query

       mask0 (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to store read/write enabled hsmp command mask0

       mask1 (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to store read/write enabled hsmp command mask1

       mask2 (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           - Input buffer to store read/write enabled hsmp command mask2

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_cpu_enabled_commands(amdsmi_processor_handle processor_handle, _Bool * r_mask, uint32_t * mask0, uint32_t * mask1, uint32_t * mask2)


.. py:function:: amdsmi_get_nic_driver_info(processor_handle)

   Retrieves information about the NIC driver

   @platform{host} @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_driver_info_t`:
               reference to the nic driver info struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_driver_info(amdsmi_processor_handle processor_handle, amdsmi_nic_driver_info_t * info)


.. py:function:: amdsmi_get_nic_asic_info(processor_handle)

   Retrieves ASIC information for the NIC

   @platform{host} @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_asic_info_t`:
               reference to the nic asic info struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_asic_info(amdsmi_processor_handle processor_handle, amdsmi_nic_asic_info_t * info)


.. py:function:: amdsmi_get_nic_bus_info(processor_handle)

   Retrieves BUS information for the NIC

   @platform{host} @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_bus_info_t`:
               reference to the nic bus info struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_bus_info(amdsmi_processor_handle processor_handle, amdsmi_nic_bus_info_t * info)


.. py:function:: amdsmi_get_nic_numa_info(processor_handle)

   Retrieves NUMA information for the NIC

   @platform{host} @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_numa_info_t`:
               reference to the nic numa info struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_numa_info(amdsmi_processor_handle processor_handle, amdsmi_nic_numa_info_t * info)


.. py:function:: amdsmi_get_nic_port_info(processor_handle)

   Retrieves PORT information for the NIC

   @platform{host} @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_port_info_t`:
               reference to the nic port info struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_port_info(amdsmi_processor_handle processor_handle, amdsmi_nic_port_info_t * info)


.. py:function:: amdsmi_get_nic_rdma_dev_info(processor_handle)

   Retrieves RDMA devices information for the NIC

   @platform{host} @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_rdma_devices_info_t`:
               reference to the nic rdma devices info struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_rdma_dev_info(amdsmi_processor_handle processor_handle, amdsmi_nic_rdma_devices_info_t * info)


.. py:function:: amdsmi_get_nic_rdma_port_statistics(processor_handle, rdma_port_index, num_stats)

   Retrieve RDMA port statistics for the NIC

   @platform{host} @platform{gpu_bm_linux}

   This function follows a two-call pattern:
   1. First call with stats=NULL to get the count of available statistics
   2. Second call with allocated array to retrieve all statistics

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

       rdma_port_index (:py:obj:`~.int`) -- *IN*:
           index of the NIC RDMA port to query

       num_stats (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           pointer to the number of statistics
           - Input: maximum number of statistics that stats array can hold
           - Output: actual number of statistics available/returned

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_stat_t`:
               pointer to array of amdsmi_nic_stat_t structures to be filled
               - If NULL, only num_stats is filled with the count of available statistics
               - If not NULL, must be allocated by user with at least num_stats elements

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_rdma_port_statistics(amdsmi_processor_handle processor_handle, uint32_t rdma_port_index, uint32_t * num_stats, amdsmi_nic_stat_t * stats)


.. py:function:: amdsmi_get_nic_fw_info(processor_handle)

   Retrieves firmware version information for the NIC

   @platform{host} @platform{gpu_bm_linux}

   Note:
       This API depends on libmnl. If libmnl is not installed on the
       system, this function returns ::AMDSMI_STATUS_NOT_SUPPORTED.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_fw_info_t`:
               reference to the nic firmware info struct.
               Must be allocated by user.

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_fw_info(amdsmi_processor_handle processor_handle, amdsmi_nic_fw_info_t * info)


.. py:function:: amdsmi_get_nic_port_statistics(processor_handle, port_index, num_stats)

   Retrieve PORT statistics for the specified NIC port

   @platform{host} @platform{gpu_bm_linux}

   This function follows a two-call pattern:
   1. First call with stats=NULL to get the count of available statistics
   2. Second call with allocated array to retrieve all statistics

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

       port_index (:py:obj:`~.int`) -- *IN*:
           index of the NIC port to query

       num_stats (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           pointer to the number of statistics
           - Input: maximum number of statistics that stats array can hold
           - Output: actual number of statistics available/returned

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_stat_t`:
               pointer to array of amdsmi_nic_stat_t structures to be filled
               - If NULL, only num_stats is filled with the count of available statistics
               - If not NULL, must be allocated by user with at least num_stats elements

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_port_statistics(amdsmi_processor_handle processor_handle, uint32_t port_index, uint32_t * num_stats, amdsmi_nic_stat_t * stats)


.. py:function:: amdsmi_get_nic_vendor_statistics(processor_handle, port_index, num_stats)

   Retrieve vendor specific statistics for the NIC port

   @platform{host} @platform{gpu_bm_linux}

   This function follows a two-call pattern:
   1. First call with stats=NULL to get the count of available statistics
   2. Second call with allocated array to retrieve all statistics

   This API provides access to vendor/driver specific statistics that may vary
   between different NIC vendors and driver/fw versions. The statistic names are
   preserved as provided by the underlying driver implementation.

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           NIC for which to query

       port_index (:py:obj:`~.int`) -- *IN*:
           index of the NIC port to query

       num_stats (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           pointer to the number of statistics
           - Input: maximum number of statistics that stats array can hold
           - Output: actual number of statistics available/returned

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
       * :py:obj:`~.amdsmi_nic_stat_t`:
               pointer to array of amdsmi_nic_stat_t structures to be filled
               - If NULL, only num_stats is filled with the count of available statistics
               - If not NULL, must be allocated by user with at least num_stats elements

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_nic_vendor_statistics(amdsmi_processor_handle processor_handle, uint32_t port_index, uint32_t * num_stats, amdsmi_nic_stat_t * stats)


.. py:class:: amdsmi_uma_carveout_option_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   UMA carveout option descriptor
       


   .. py:attribute:: index
      :type:  Any


   .. py:attribute:: description
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_uma_carveout_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   UMA carveout configuration information
       


   .. py:attribute:: current_index
      :type:  Any


   .. py:attribute:: num_options
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: amdsmi_ttm_info_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   TTM (Translation Table Manager) configuration information
       


   .. py:attribute:: current_pages
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: amdsmi_get_gpu_uma_carveout_info(processor_handle)

   Get UMA carveout configuration information

   This function retrieves the current UMA (Unified Memory Architecture) carveout
   configuration for the specified GPU. UMA carveout controls dedicated GPU memory
   allocation on APU systems.

   Note:
       This uses a kernel UAPI sysfs interface, not libdrm.

   @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           GPU device handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success
           ::AMDSMI_STATUS_NOT_SUPPORTED if UMA carveout is not available on this device
           ::AMDSMI_STATUS_INVAL if info is nullptr
       * :py:obj:`~.amdsmi_uma_carveout_info_t`:
               Pointer to receive UMA carveout information

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_gpu_uma_carveout_info(amdsmi_processor_handle processor_handle, amdsmi_uma_carveout_info_t * info)


.. py:function:: amdsmi_set_gpu_uma_carveout(processor_handle, option_index)

   Set UMA carveout configuration

   This function sets the UMA carveout configuration for the specified GPU.
   The system must be rebooted for changes to take effect.

   Note:
       This uses a kernel UAPI sysfs interface, not libdrm.

   @platform{gpu_bm_linux}

   Args:
       processor_handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           GPU device handle

       option_index (:py:obj:`~.int`) -- *IN*:
           Index of the carveout option to set

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success
           ::AMDSMI_STATUS_NOT_SUPPORTED if UMA carveout is not available on this device
           ::AMDSMI_STATUS_NO_PERM if insufficient permissions
           ::AMDSMI_STATUS_INVAL if option_index is out of range

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_gpu_uma_carveout(amdsmi_processor_handle processor_handle, uint32_t option_index)


.. py:function:: amdsmi_get_ttm_info()

   Get TTM configuration information

   This function retrieves the current TTM (Translation Table Manager) pages limit.
   TTM controls shared GPU memory (GTT) allocation.

   Note:
       This uses a kernel UAPI interface (modprobe.d), not libdrm.

   @platform{gpu_bm_linux}

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success
           ::AMDSMI_STATUS_NOT_SUPPORTED if TTM configuration is not available
           ::AMDSMI_STATUS_INVAL if info is nullptr
       * :py:obj:`~.amdsmi_ttm_info_t`:
               Pointer to receive TTM configuration information

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_get_ttm_info(amdsmi_ttm_info_t * info)


.. py:function:: amdsmi_set_ttm_pages_limit(pages)

   Set TTM pages limit

   This function configures the TTM pages limit by creating/updating
   /etc/modprobe.d/ttm.conf. The system must be rebooted for changes to take effect.

   Note:
       This uses a kernel UAPI interface (modprobe.d), not libdrm.

   @platform{gpu_bm_linux}

   Args:
       pages (:py:obj:`~.int`) -- *IN*:
           Number of pages to allocate for TTM

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success
           ::AMDSMI_STATUS_NO_PERM if insufficient permissions
           ::AMDSMI_STATUS_INVAL if pages is 0

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_set_ttm_pages_limit(uint64_t pages)


.. py:function:: amdsmi_reset_ttm_pages_limit()

   Reset TTM pages limit to system default

   This function resets the TTM pages limit to system default by removing
   /etc/modprobe.d/ttm.conf. The system must be rebooted for changes to take effect.

   Note:
       This uses a kernel UAPI interface (modprobe.d), not libdrm.

   @platform{gpu_bm_linux}

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amdsmi_status_t`: amdsmi_status_t
           ::AMDSMI_STATUS_SUCCESS on success
           ::AMDSMI_STATUS_NO_PERM if insufficient permissions

   .. rubric:: C signature

   .. code-block:: c

       amdsmi_status_t amdsmi_reset_ttm_pages_limit()


