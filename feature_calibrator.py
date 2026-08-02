import numpy as np

class FeatureCalibrator:
    """
    Feature Calibration & Aggregation Engine for Phase 2.
    Converts continuous frame-level visual detections (counts, depth meters, TTC near-misses)
    into the exact 10 integer scales required by the XGBoost Risk Classifier model.
    """
    
    @staticmethod
    def calibrate_segment_metrics(aggregated_counts: dict) -> dict:
        """
        Input: Raw aggregated counts/metrics over a 100m-500m video segment.
        Output: Dictionary matching the 10-parameter model payload schema.
        """
        # 1. Sight Distance Adequacy (0: Adequate > 100m, 1: Partly 50-100m, 2: Blind < 50m)
        sight_meters = aggregated_counts.get("avg_sight_distance_m", 120.0)
        if sight_meters >= 100.0:
            sight_dist_ord = 0
        elif sight_meters >= 50.0:
            sight_dist_ord = 1
        else:
            sight_dist_ord = 2
            
        # 2. Roadside Commercial Density (0: None, 2: Low 1-2, 3: Medium 3-5, 5: High > 5)
        comm_count = aggregated_counts.get("commercial_count", 0)
        if comm_count == 0:
            comm_ord = 0
        elif comm_count <= 2:
            comm_ord = 2
        elif comm_count <= 5:
            comm_ord = 3
        else:
            comm_ord = 5
            
        # 3. Pedestrian-Vehicle Conflicts (0: None, 2: Few 1-3, 5: Frequent > 3)
        ped_conflicts = aggregated_counts.get("pedestrian_conflicts", 0)
        if ped_conflicts == 0:
            ped_conflict_ord = 0
        elif ped_conflicts <= 3:
            ped_conflict_ord = 2
        else:
            ped_conflict_ord = 5
            
        # 4. Frequency of Jaywalking Pedestrians (0: None, 2: Occasional 1-3, 5: Frequent > 3)
        jaywalking_count = aggregated_counts.get("jaywalking_count", 0)
        if jaywalking_count == 0:
            jaywalk_ord = 0
        elif jaywalking_count <= 3:
            jaywalk_ord = 2
        else:
            jaywalk_ord = 5
            
        # 5. Unsafe Merging/Diverging (0: None, 2: Few 1-2, 5: Frequent > 2)
        unsafe_maneuvers = aggregated_counts.get("unsafe_merging_count", 0)
        if unsafe_maneuvers == 0:
            unsafe_ord = 0
        elif unsafe_maneuvers <= 2:
            unsafe_ord = 2
        else:
            unsafe_ord = 5
            
        # 6. Visibility of Pedestrian Crossings (0: Good, 1: Poor, 2: Absent)
        crosswalk_vis = aggregated_counts.get("crosswalk_visibility_code", 0)
        crosswalk_vis_ord = int(np.clip(crosswalk_vis, 0, 2))
        
        # 7. Road Surface Condition (0: Good, 1: Minor defects, 2: Damaged)
        surface_defects = aggregated_counts.get("surface_defect_code", 0)
        surface_ord = int(np.clip(surface_defects, 0, 2))
        
        # 8. Lane Marking Condition (0: Clear, 1: Faded, 2: Absent)
        lane_marking = aggregated_counts.get("lane_marking_code", 0)
        lane_marking_ord = int(np.clip(lane_marking, 0, 2))
        
        # 9. Junction Channelization (0: Proper, 1: Poor, 2: Absent)
        channelization = aggregated_counts.get("channelization_code", 0)
        channelization_ord = int(np.clip(channelization, 0, 2))
        
        # 10. Presence of Animals on Road (0: None, 1: Rare 1, 5: Frequent > 1)
        animal_count = aggregated_counts.get("animal_count", 0)
        if animal_count == 0:
            animal_ord = 0
        elif animal_count == 1:
            animal_ord = 1
        else:
            animal_ord = 5
            
        return {
            'Is sight distance adequate for safe driving?_ord': sight_dist_ord,
            'Roadside commercial activity density_ord': comm_ord,
            'Pedestrian-vehicle conflicts_ord': ped_conflict_ord,
            'Frequency of jaywalking pedestrians_ord': jaywalk_ord,
            'Unsafe merging diverging maneuvers observed_ord': unsafe_ord,
            'Visibility of pedestrian crossings_ord': crosswalk_vis_ord,
            'Road surface condition_ord': surface_ord,
            'Lane marking condition_ord': lane_marking_ord,
            'Junction channelization_ord': channelization_ord,
            'Presence of animals on road_ord': animal_ord
        }


if __name__ == "__main__":
    test_raw_metrics = {
        "avg_sight_distance_m": 45.0, # Blind sight distance
        "commercial_count": 4,        # Medium commercial density
        "pedestrian_conflicts": 4,    # Frequent conflicts
        "jaywalking_count": 2,        # Occasional jaywalking
        "animal_count": 2             # Frequent animals
    }
    
    calibrated = FeatureCalibrator.calibrate_segment_metrics(test_raw_metrics)
    print("Calibrated 10-Parameter XGBoost Input Payload:")
    for k, v in calibrated.items():
        print(f"  • {k}: {v}")
