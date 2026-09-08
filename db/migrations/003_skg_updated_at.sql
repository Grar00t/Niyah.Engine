-- Database-managed modification timestamps for canonical SKG objects.
-- This is intentionally separate from application/runtime policy.

create or replace function skg.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_skg_nodes_touch_updated_at
    on skg.nodes;

create trigger trg_skg_nodes_touch_updated_at
before update on skg.nodes
for each row
execute function skg.touch_updated_at();

drop trigger if exists trg_skg_edges_touch_updated_at
    on skg.edges;

create trigger trg_skg_edges_touch_updated_at
before update on skg.edges
for each row
execute function skg.touch_updated_at();
