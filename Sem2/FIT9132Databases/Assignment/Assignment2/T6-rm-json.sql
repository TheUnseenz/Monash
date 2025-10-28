/*****PLEASE ENTER YOUR DETAILS BELOW*****/
--T6-rm-json.sql

--Student ID: 27030768
--Student Name: Adrian Leong Tat Wei


/* Comments for your marker:




*/


-- PLEASE PLACE REQUIRED SQL SELECT STATEMENT FOR THIS PART HERE
-- ENSURE that your query is formatted and has a semicolon
-- (;) at the end of this answer

SET PAGESIZE 0 -- Prevent headers being printed 

SELECT
JSON_SERIALIZE(
    JSON_OBJECT(
        '_id' VALUE t.team_id,
        'carn_name' VALUE c.carn_name,
        'carn_date' VALUE TO_CHAR(c.carn_date, 'DD-Mon-YYYY'),
        'team_name' VALUE t.team_name,
        'team_leader' VALUE JSON_OBJECT(
            'name' VALUE comp_leader.comp_fname || ' ' || comp_leader.comp_lname,
            'phone' VALUE NVL(comp_leader.comp_phone, '-'),
            'email' VALUE NVL(comp_leader.comp_email, '-')
        ),
        'team_no_of_members' VALUE (
            SELECT COUNT(e_mem.entry_no)
            FROM ENTRY e_mem
            WHERE e_mem.team_id = t.team_id
            AND e_mem.event_id IN (SELECT event_id FROM EVENT WHERE carn_date = t.carn_date)
        ),
        'team_members' VALUE (
            SELECT
                JSON_ARRAYAGG(
                    JSON_OBJECT(
                        'competitor_name' VALUE comp_member.comp_fname || ' ' || comp_member.comp_lname,
                        'competitor_phone' VALUE NVL(comp_member.comp_phone, '-'),
                        'event_type' VALUE et_member.eventtype_desc,
                        'entry_no' VALUE e_member.entry_no,
                        'starttime' VALUE NVL(TO_CHAR(e_member.entry_starttime, 'HH24:MI:SS'), '-'),
                        'finishtime' VALUE NVL(TO_CHAR(e_member.entry_finishtime, 'HH24:MI:SS'), '-'),
                        'elapsedtime' VALUE NVL(TO_CHAR(e_member.entry_elapsedtime, 'HH24:MI:SS'), '-')
                    )
                    ORDER BY comp_member.comp_lname, comp_member.comp_fname
                )
            FROM
                ENTRY e_member
            JOIN
                COMPETITOR comp_member ON e_member.comp_no = comp_member.comp_no
            JOIN
                EVENT ev_member ON e_member.event_id = ev_member.event_id
            JOIN
                EVENTTYPE et_member ON ev_member.eventtype_code = et_member.eventtype_code
            WHERE
                e_member.team_id = t.team_id
            AND e_member.event_id IN (SELECT event_id FROM EVENT WHERE carn_date = t.carn_date)
        )
    )
    PRETTY -- Format json output for readability
) AS TEAM_JSON_DOCUMENT
FROM
    TEAM t
JOIN
    CARNIVAL c ON t.carn_date = c.carn_date
JOIN
    ENTRY e_leader ON t.event_id = e_leader.event_id AND t.entry_no = e_leader.entry_no
JOIN
    COMPETITOR comp_leader ON e_leader.comp_no = comp_leader.comp_no
ORDER BY
    t.team_id;



