#!/usr/bin/env python3

class MarkerHandler:
    def __init__(self):
        '''
            Initializes the MarkerHandler class.
            This class is responsible for managing the observed ArUco markers and their positions.
        '''
        # Dictionary to hold the positions of the markers, with the ArUco ID as the key.
        self.markers = {}

        # Dictionary to keep track of observed ArUco IDs and their corresponding indices
        # so as to avoid duplicate ArUco IDs scan.
        self.observed_arucos = {}

    
    def add_marker(self, aruco_id, pose_world):
        aruco_id = int(aruco_id)
        if aruco_id not in self.observed_arucos:
            self.observed_arucos[aruco_id] = len(self.observed_arucos)
            self.markers[aruco_id] = pose_world


    def get_index(self, aruco_id):
        """
        Retrieves the index of the given ArUco marker.
        
        :param aruco_id: The ID of the ArUco marker.
        :return: The index of the ArUco marker, or None if it is not found.
        """
        # Return the index of the ArUco marker from the observed_arucos dictionary.
        return self.observed_arucos.get(int(aruco_id), None)
    

    def get_marker_position(self, aruco_id):
        """
        Retrieves the position of the given ArUco marker.

        :param aruco_id: The ID of the ArUco marker.
        :return: The position of the ArUco marker, or None if it is not found.
        """
        # Return the position of the ArUco marker from the markers dictionary.
        return self.markers.get(int(aruco_id), None)