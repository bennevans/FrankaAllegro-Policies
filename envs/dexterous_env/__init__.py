from gym.envs.registration import register 


register(
	id='CupUnstacking-v1',
	entry_point='dexterous_env.cup_unstacking_env:CupUnstackingEnv',
	max_episode_steps=76,
)

register(
	id='BowlUnstacking-v1',
	entry_point='dexterous_env.bowl_unstacking_env:BowlUnstackingEnv',
	max_episode_steps=76,
)

register(
    id = 'PlierPicking-v1',
    entry_point='dexterous_env.plier_picking_env:PlierPicking',
	max_episode_steps=80,
)

register(
    id = 'SpongeFlipping-v1',
    entry_point='dexterous_env.sponge_flipping_env:SpongeFlipping',
	max_episode_steps=80,
)

register(
    id = 'SpongeFlipping-v2',
    entry_point='dexterous_env.sponge_flipping_env:SpongeFlippingCurvedFranka',
    max_episode_steps=80
)


register(
    id = 'CardTurning-v1',
    entry_point='dexterous_env.card_turning_env:CardTurning',
	max_episode_steps=80,
)

register(
    id = 'PegInsertion-v1',
    entry_point='dexterous_env.peg_insertion_env:PegInsertion',
	max_episode_steps=80,
)

register(
    id = 'MintOpening-v1',
    entry_point='dexterous_env.mint_opening_env:MintOpening',
    max_episode_steps=80
)

register(
    id = 'MultiTask-v1',
    entry_point='dexterous_env.multi_task_env:MultiTask',
    max_episode_steps=240
)

register(
    id = 'PlierClipping-v1',
    entry_point='dexterous_env.plier_clipping_env:PlierClipping',
    max_episode_steps=80
)

register(
    id = 'MoveFranka-v1',
    entry_point='dexterous_env.move_franka_env:MoveFranka',
    max_episode_steps=80
)

register(
    id = 'StateRecognitionR1B0R',
    entry_point='dexterous_env.state_recognition_r1b0r_env:StateRecognitionR1B0R',
    max_episode_steps=80
)

register(
    id = 'GeneralizationPinchGrasp-v1',
    entry_point='dexterous_env.generalization_pinch_grasp_env:GeneralizationPinchGrasp',
    max_episode_steps=80
)

register(
    id = 'HondaFingerGait-v1',
    entry_point='dexterous_env.finger_gaiting:FingerGaitingCurvedFranka',
    max_episode_steps=80
)

register(
    id = 'SpongeSliding-v1',
    entry_point='dexterous_env.sponge_sliding:SpongeSliding',
    max_episode_steps=80
)

register(
    id = 'ObjectRotate-v1',
    entry_point='dexterous_env.object_rotate:ObjectRotate',
    max_episode_steps=80
)

register(
    id = 'MusicBoxOpen-v1',
    entry_point='dexterous_env.box_opening:MusicBoxOpening',
    max_episode_steps=80
)

register(
    id = 'TeaPicking-v1', 
    entry_point='dexterous_env.tea_picking_env:TeaPicking',
    max_episode_steps=80
)
