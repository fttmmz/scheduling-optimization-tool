-- Apply after deploying the isolated backend database client with SUPABASE_SERVER_KEY
-- and verifying login, instructor preferences, and admin actions.
begin;

revoke select, insert, update, delete, truncate on table
  public."college",
  public."course_req",
  public."courses",
  public."department",
  public."instructor",
  public."preference",
  public."registrar_practice",
  public."room",
  public."rule_set",
  public."schedule",
  public."schedule_detailes",
  public."scheduling_rule",
  public."section_info",
  public."student",
  public."timeslot",
  public."user"
from PUBLIC, anon, authenticated;

drop policy if exists "Full access college" on public."college";
drop policy if exists "Full access course_req" on public."course_req";
drop policy if exists "Full access courses" on public."courses";
drop policy if exists "Full access department" on public."department";
drop policy if exists "Full access instructor" on public."instructor";
drop policy if exists "Full access preference" on public."preference";
drop policy if exists "registrar_practice readable" on public."registrar_practice";
drop policy if exists "Full access room" on public."room";
drop policy if exists "Full access rule_set" on public."rule_set";
drop policy if exists "Full access schedule" on public."schedule";
drop policy if exists "Full access schedule_detailes" on public."schedule_detailes";
drop policy if exists "Full access scheduling_rule" on public."scheduling_rule";
drop policy if exists "read section_info" on public."section_info";
drop policy if exists "Full access student" on public."student";
drop policy if exists "Full access timeslot" on public."timeslot";
drop policy if exists "Full access user" on public."user";

commit;
