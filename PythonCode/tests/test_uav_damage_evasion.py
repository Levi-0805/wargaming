import math
from test_three_way_recon import recon_team, at_stations
from test_fslutk_rule import state,entity


def test_damage_changes_station_with_bounded_goal_and_retains_sensor_range():
    objects,units,soldiers,route,control,current=recon_team(5)
    for u in units:control.choose(current,u,objects[0],route)
    at_stations(objects,units,control)
    u=units[0]
    control.choose(state(units,objectives=objects,step=1),u,objects[0],route)
    original=tuple(u.position);u.hp=50
    s=state(units,objectives=objects,step=2)
    action=control.choose(s,u,objects[0],route)
    assert action.command=='Moving' and action.points!=original
    point=control._evasion[u.uid][1]
    assert math.dist(point,objects[0].position)<u.properties['perceptionRange']
    for i in range(3,8):
        u.hp-=1
        control.choose(state(units,objectives=objects,step=i),u,objects[0],route)
        assert control._evasion[u.uid][1]==point
    assert control._split_observer(s,u,objects[0])


def test_sighting_without_damage_does_not_trigger_escape():
    objects,units,soldiers,route,control,current=recon_team(5)
    for u in units:control.choose(current,u,objects[0],route)
    at_stations(objects,units,control)
    enemy=entity(999,1,0,*objects[0].position[:2]);u=units[0]
    u.raw['agentPerception']=[999]
    for i in range(1,5):control.choose(state(units,[enemy],objects,step=i),u,objects[0],route)
    assert not control._evasion


def test_last_three_aircraft_keep_recon_roles_after_surveys():
    objects,units,soldiers,route,control,current=recon_team(5)
    control.prepare_recon(current,route);at_stations(objects,units,control)
    for u in units[3:]:u.alive=False
    for i in range(1,5):
        s=state(units,objectives=objects,step=i);control.prepare_recon(s,route)
    for u in units[:3]:
        target=s.key_object_by_uid(control._split_objectives[u.uid])
        assert control._split_observer(s,u,target)


def test_new_episode_discards_old_damage_and_escape_goal():
    objects,units,soldiers,route,control,current=recon_team(5)
    u=units[0];control.choose(current,u,objects[0],route);u.hp=50
    control.choose(state(units,objectives=objects,step=1),u,objects[0],route)
    assert control._evasion
    control.choose(state(units,objectives=objects,episode=2),u,objects[0],route)
    assert not control._evasion


def test_nearby_locally_detected_enemy_uav_triggers_evasion_before_damage():
    objects,units,soldiers,route,control,current=recon_team(5)
    control.prepare_recon(current,route);at_stations(objects,units,control)
    u=units[0];enemy=entity(999,1,0,u.position[0]+2000,u.position[1],kind='BP_Base_UAV_C')
    enemy.position[2]=u.position[2];u.raw['agentPerception']=[999]
    action=control.choose(state(units,[enemy],objects,step=1),u,objects[0],route)
    assert action.command=='Moving' and u.uid in control._evasion


def test_shared_enemy_uav_does_not_trigger_blind_evasion():
    objects,units,soldiers,route,control,current=recon_team(5)
    control.prepare_recon(current,route);at_stations(objects,units,control)
    u=units[0];enemy=entity(999,1,0,*u.position[:2],kind='BP_Base_UAV_C')
    enemy.position[2]=u.position[2];u.raw['agentPerception']=[]
    control.choose(state(units,[enemy],objects,step=1),u,objects[0],route)
    assert not control._evasion
