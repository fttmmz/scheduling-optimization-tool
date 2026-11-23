-- Run against the non-production Supabase project first.
-- The timetable application has no workflow that truncates tables as an API role.
begin;

do $$
declare
  target record;
begin
  for target in
    select c.oid, n.nspname, c.relname
    from pg_catalog.pg_class c
    join pg_catalog.pg_namespace n on n.oid = c.relnamespace
    where n.nspname = 'public'
      and c.relkind in ('r', 'p')
      and c.relname in (
        'building_import', 'college', 'course_req', 'courses', 'department',
        'dept_mapping', 'instructor', 'preference', 'registrar_practice', 'room',
        'rule_set', 'schedule', 'schedule_detailes', 'scheduling_rule',
        'section_info', 'student', 'timeslot', 'user'
      )
  loop
    execute format(
      'revoke truncate on table %I.%I from public, anon, authenticated',
      target.nspname, target.relname
    );

    if pg_catalog.has_table_privilege('anon', target.oid, 'TRUNCATE')
       or pg_catalog.has_table_privilege('authenticated', target.oid, 'TRUNCATE') then
      raise exception 'An API role still inherits TRUNCATE on %.%',
        target.nspname, target.relname;
    end if;
  end loop;
end;
$$;

commit;
