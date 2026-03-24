dataset_info = dict(
    dataset_name='panoramic_teeth_structured_v2',
    paper_info=dict(
        author='Local dataset',
        title='Panoramic Teeth Structured Contour and Keypoint Dataset',
        year='2026',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(name='M_C', id=0, color=[0, 178, 255], type='upper', swap='D_C'),
        1:
        dict(name='M_B', id=1, color=[0, 178, 255], type='upper', swap='D_B'),
        2:
        dict(name='A', id=2, color=[255, 214, 10], type='lower', swap='A'),
        3:
        dict(name='D_B', id=3, color=[255, 82, 82], type='upper', swap='M_B'),
        4:
        dict(name='D_C', id=4, color=[255, 82, 82], type='upper', swap='M_C'),
    },
    skeleton_info={
        0: dict(link=('M_C', 'M_B'), id=0, color=[0, 178, 255]),
        1: dict(link=('M_B', 'A'), id=1, color=[0, 178, 255]),
        2: dict(link=('A', 'D_B'), id=2, color=[255, 82, 82]),
        3: dict(link=('D_B', 'D_C'), id=3, color=[255, 82, 82]),
    },
    joint_weights=[1., 1., 1., 1., 1.],
    sigmas=[0.025, 0.025, 0.03, 0.025, 0.025],
)

