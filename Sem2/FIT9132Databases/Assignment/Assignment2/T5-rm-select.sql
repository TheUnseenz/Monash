/*****PLEASE ENTER YOUR DETAILS BELOW*****/
--T5-rm-select.sql

--Student ID: 27030768
--Student Name: Adrian Leong Tat Wei


/* Comments for your marker:




*/


/* (a) Most popular team name report */
-- PLEASE PLACE REQUIRED SQL SELECT STATEMENT FOR THIS PART HERE
-- ENSURE that your query is formatted and has a semicolon
-- (;) at the end of this answer
SELECT t.team_name AS "TEAM_NAME",
       to_char(
           c.carn_date,
           'DD-Mon-YYYY'
       ) AS "CARNIVAL_DATE",
       comp_leader.comp_fname
       || ' '
       || comp_leader.comp_lname AS "TEAMLEADER",
       (
           SELECT COUNT(e_member.entry_no)
             FROM entry e_member
            WHERE e_member.team_id = t.team_id
              AND e_member.event_id IN ( -- Ensure team members are from the same carnival as the team itself
               SELECT event_id
                 FROM event
                WHERE carn_date = t.carn_date
           )
       ) AS "TEAM_NO_MEMBERS"
  FROM team t
  JOIN carnival c
ON t.carn_date = c.carn_date
  JOIN entry e_leader
ON t.event_id = e_leader.event_id
   AND t.entry_no = e_leader.entry_no
  JOIN competitor comp_leader
ON e_leader.comp_no = comp_leader.comp_no
 WHERE t.team_name IN (
    SELECT team_name
      FROM team
     WHERE carn_date < TO_DATE('9-JUN-2025',
            'DD-MON-YYYY') -- Completed carnivals are in the past
     GROUP BY team_name
    HAVING COUNT(*) = (
        SELECT MAX(name_count) -- Most popular team name(s), so just the most popular and not descending order
          FROM (
            SELECT COUNT(*) AS name_count
              FROM team
             WHERE carn_date < TO_DATE('9-JUN-2025',
            'DD-MON-YYYY') -- Completed carnivals are in the past
             GROUP BY team_name
        )
    )
)
   AND c.carn_date < TO_DATE('9-JUN-2025','DD-MON-YYYY') -- Ensure only completed carnivals
 ORDER BY t.team_name,
          c.carn_date;


/* (b) Current record holder by event type */
-- PLEASE PLACE REQUIRED SQL SELECT STATEMENT FOR THIS PART HERE
-- ENSURE that your query is formatted and has a semicolon
-- (;) at the end of this answer
SELECT et.eventtype_desc AS "Event",
       c.carn_name
       || ', '
       || to_char(
           c.carn_date,
           'Day'
       )
       || ' '
       || to_char(
           c.carn_date,
           'DD-Mon-YYYY'
       ) AS "Carnival",
       to_char(
           e.entry_elapsedtime,
           'HH24:MI:SS'
       ) AS "Current Record",
       comp.comp_no
       || ' - '
       || comp.comp_fname
       || ' '
       || comp.comp_lname AS "Competitor No and Name",
       trunc(months_between(
           c.carn_date,
           comp.comp_dob
       ) / 12) AS "Age at Carnival"
  FROM entry e
  JOIN event ev
ON e.event_id = ev.event_id
  JOIN eventtype et
ON ev.eventtype_code = et.eventtype_code
  JOIN carnival c
ON ev.carn_date = c.carn_date
  JOIN competitor comp
ON e.comp_no = comp.comp_no
 WHERE e.entry_elapsedtime IS NOT NULL
   AND ( et.eventtype_code,
         e.entry_elapsedtime ) IN (
    SELECT ev2.eventtype_code,
           MIN(e2.entry_elapsedtime)
      FROM entry e2
      JOIN event ev2
    ON e2.event_id = ev2.event_id
     WHERE e2.entry_elapsedtime IS NOT NULL
     GROUP BY ev2.eventtype_code
)
 ORDER BY et.eventtype_desc,
          comp.comp_no;


/* (c) Entries by carnival and event type*/
-- PLEASE PLACE REQUIRED SQL SELECT STATEMENT FOR THIS PART HERE
-- ENSURE that your query is formatted and has a semicolon
-- (;) at the end of this answer
SELECT c.carn_name AS "Carnival Name",
       to_char(
           c.carn_date,
           'DD-Mon-YYYY'
       ) AS "Carnival Date",
       et.eventtype_desc AS "Event Description",
       CASE
           WHEN COUNT(e.entry_no) = 0 THEN
               rpad(
                   'Not offered',
                   17,
                   ' '
               ) -- Left-aligned 'Not offered'
           ELSE
               lpad(
                   to_char(count(e.entry_no)),
                   17,
                   ' '
               ) -- Right-aligned number
       END AS "Number of Entries",
       lpad(
           CASE
               WHEN SUM(count(e.entry_no))
                    OVER(PARTITION BY c.carn_date) = 0 THEN
                   to_char(0)
               ELSE to_char(trunc(COUNT(e.entry_no) * 100 / SUM(count(e.entry_no))
                                                            OVER(PARTITION BY c.carn_date
                                                            )))
           END,
           21,
           ' '
       ) AS "% of Carnival Entries"
  FROM carnival c
 CROSS JOIN eventtype et
  LEFT JOIN event ev
ON c.carn_date = ev.carn_date
   AND et.eventtype_code = ev.eventtype_code
  LEFT JOIN entry e
ON ev.event_id = e.event_id
 GROUP BY c.carn_name,
          c.carn_date,
          et.eventtype_desc
 ORDER BY c.carn_date,
          COUNT(e.entry_no) DESC,
          et.eventtype_desc;