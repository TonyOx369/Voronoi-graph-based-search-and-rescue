#include <project_finder/FinderBot.hpp>


FinderBot::FinderBot(const std::string& ns) : namespace_(ns) {
    // initialize send goal state to false
    send_goal_state_ = false;

    // initialize mission accomplished status to false
    mission_accomplished_ = false;

    // initialize mission completed count to 0
    mission_count_ = 0;
}

FinderBot::~FinderBot() {
}

std::string FinderBot::get_namespace() {
    return namespace_;
}

void FinderBot::set_waypoint(Pose waypoint) {
    // set waypoint
    waypoint_ = waypoint;
    // set default destination to waypoint
    destination_ = waypoint_;
}

Pose FinderBot::get_waypoint() {
    return waypoint_;
}

void FinderBot::set_fire_exit(Pose fire_exit) {
    fire_exit_ = fire_exit;
}

Pose FinderBot::get_fire_exit() {
    return fire_exit_;
}

Pose FinderBot::get_destination() {
    return destination_;
}

void FinderBot::update_destination(Pose dest) {
    destination_ = dest;
}

bool FinderBot::get_send_goal_state() {
    return send_goal_state_;
}

void FinderBot::set_send_goal_state(bool send_goal) {
    send_goal_state_ = send_goal;
}

bool FinderBot::get_mission_status() {
    return mission_accomplished_;
}

void FinderBot::set_mission_status(bool status) {
    mission_accomplished_ = status;
}

void FinderBot::set_home_location(Pose home_sweet_home) {
    home_location_ = home_sweet_home;
}

Pose FinderBot::get_home_location() {
    return home_location_;
}

void FinderBot::increment_mission_count() {
    mission_count_ += 1;
}

int FinderBot::get_mission_count() {
    return mission_count_;
}
