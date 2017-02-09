import os
import tempfile

if __name__ == "__main__":
    import cytomine

    # Connect to cytomine, edit connection values
    cytomine_host = "demo.cytomine.be"
    cytomine_public_key = "XXX"  # to complete
    cytomine_private_key = "XXX"  # to complete
    id_project = -1  # to complete

    # Connection to Cytomine Core
    conn = cytomine.Cytomine(
        cytomine_host,
        cytomine_public_key,
        cytomine_private_key,
        base_path='/api/',
        working_path=os.path.join(tempfile.gettempdir(), "cytomine"),
        verbose=True
    )

    # define software parameter template
    software = conn.add_software("Demo_SLDC_Workflow_With_Pyxit", "pyxitSuggestedTermJobService", "ValidateAnnotation")
    conn.add_software_parameter("cytomine_id_software", software.id, "Number", 0, True, 1, True)
    conn.add_software_parameter("cytomine_id_project", software.id, "Number", 0, True, 100, True)
    conn.add_software_parameter("cytomine_id_image", software.id, "Number", 0, True, 200, True)
    conn.add_software_parameter("n_jobs", software.id, "Number", 1, True, 300, False)
    conn.add_software_parameter("min_area", software.id, "Number", 12, True, 400, False)
    conn.add_software_parameter("threshold", software.id, "Number", 140, True, 500, False)
    conn.add_software_parameter("sldc_tile_overlap", software.id, "Number", 10, True, 600, False)
    conn.add_software_parameter("sldc_tile_width", software.id, "Number", 768, True, 700, False)
    conn.add_software_parameter("sldc_tile_height", software.id, "Number", 768, True, 800, False)
    conn.add_software_parameter("pyxit_model_path", software.id, "Number", "", True, 900, False)
    conn.add_software_parameter("n_jobs", software.id, "Number", 1, True, 1000, False)
    conn.add_software_parameter("min_area", software.id, "Number", 500, True, 1100, False)
    conn.add_software_parameter("threshold", software.id, "Number", 215, True, 1200, False)
    conn.add_software_parameter("rseed", software.id, "Number", 0, True, 1300, False)
    conn.add_software_parameter("working_path", software.id, "Number", "", True, 1400, False)

    # add software to a given project
    addSoftwareProject = conn.add_software_project(id_project, software.id)
