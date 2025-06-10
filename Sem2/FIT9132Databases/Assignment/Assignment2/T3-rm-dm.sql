--****PLEASE ENTER YOUR DETAILS BELOW****
--T3-rm-dm.sql

--Student ID:
--Student Name:

/* Comments for your marker:




*/

--(a) Create Sequences
-- Drop sequence for COMPETITOR if it exists
DROP SEQUENCE competitor_seq;

-- Create sequence for COMPETITOR
CREATE SEQUENCE competitor_seq
START WITH 100
INCREMENT BY 5;

-- Drop sequence for TEAM if it exists
DROP SEQUENCE team_seq;

-- Create sequence for TEAM
CREATE SEQUENCE team_seq
START WITH 100
INCREMENT BY 5;


-- (b) Record New Competitors, Team, and Entries
-- Re-ordered to ensure parent keys exist before children are inserted

-- Insert Keith Rose
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
VALUES (competitor_seq.NEXTVAL, 'Keith', 'Rose', 'M', TO_DATE('15-JAN-1990', 'DD-MON-YYYY'), 'keith.rose@monash.edu', 'Y', '0422141112');

-- Insert Jackson Bull
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
VALUES (competitor_seq.NEXTVAL, 'Jackson', 'Bull', 'M', TO_DATE('20-MAR-1992', 'DD-MON-YYYY'), 'jackson.bull@monash.edu', 'Y', '0422412524');

-- Insert Keith's entry (as team leader) FIRST, so its entry_no is available for the TEAM table.
-- Using MAX(entry_no) + 1 for Keith's entry_no for the 10 Km Run event.
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
SELECT
    e.event_id,
    NVL(MAX(ent.entry_no), 0) + 1, -- Calculate next entry_no for this event_id
    TO_TIMESTAMP('08:30:00', 'HH24:MI:SS'), NULL, NULL,
    (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112'),
    NULL, -- Team_id will be updated after team creation
    (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')
FROM EVENT e
JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
LEFT JOIN ENTRY ent ON e.event_id = ent.event_id -- Join to ENTRY to get MAX(entry_no)
WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
  AND et.eventtype_desc = '10 Km Run'
GROUP BY e.event_id; -- Group by event_id to get MAX for that specific event_id

-- Now insert the 'Super Runners' team, referencing Keith's newly created entry.
-- This requires knowing Keith's comp_no and the entry_no assigned to him.
-- We will re-select the entry_no for Keith's specific entry.
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no)
SELECT
    team_seq.NEXTVAL,
    'Super Runners',
    c.carn_date,
    e.event_id,
    ent.entry_no -- This is Keith's entry_no for the 10km run
FROM CARNIVAL c
JOIN EVENT e ON c.carn_date = e.carn_date
JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
JOIN ENTRY ent ON e.event_id = ent.event_id
JOIN COMPETITOR comp ON ent.comp_no = comp.comp_no
WHERE c.carn_name = 'RM Winter Series Caulfield 2025'
  AND et.eventtype_desc = '10 Km Run'
  AND comp.comp_phone = '0422141112';


-- Update Keith's entry to set team_id (must be done after team insert)
UPDATE ENTRY
SET team_id = (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'))
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112')
  AND event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025') AND et.eventtype_desc = '10 Km Run');

-- Insert Jackson's entry for the 10 Km Run event.
-- Using MAX(entry_no) + 1 for Jackson's entry_no for the same event.
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
SELECT
    e.event_id,
    NVL(MAX(ent.entry_no), 0) + 1, -- Calculate next entry_no for this event_id
    TO_TIMESTAMP('08:30:05', 'HH24:MI:SS'), NULL, NULL,
    (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524'),
    (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')),
    (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')
FROM EVENT e
JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
LEFT JOIN ENTRY ent ON e.event_id = ent.event_id -- Join to ENTRY to get MAX(entry_no)
WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
  AND et.eventtype_desc = '10 Km Run'
GROUP BY e.event_id; -- Group by event_id to get MAX for that specific event_id

COMMIT;


-- (c) Update Jackson Bull's Entry
-- Delete Jackson's old entry for the 10 Km Run
DELETE FROM ENTRY
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524')
  AND event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025') AND et.eventtype_desc = '10 Km Run');

-- Insert Jackson's new entry for the 5 Km Run (as individual runner)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
SELECT
    e.event_id,
    NVL(MAX(ent.entry_no), 0) + 1, -- Calculate next entry_no for this event_id
    TO_TIMESTAMP('08:45:00', 'HH24:MI:SS'), NULL, NULL,
    (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524'),
    NULL, -- No team for individual runner
    (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue')
FROM EVENT e
JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
LEFT JOIN ENTRY ent ON e.event_id = ent.event_id -- Join to ENTRY to get MAX(entry_no)
WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
  AND et.eventtype_desc = '5 Km Run'
GROUP BY e.event_id;
COMMIT;


-- (d) Keith Withdraws, Team Disbands
-- Update Jackson's entry to remove him from the team
UPDATE ENTRY
SET team_id = NULL
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524')
  AND team_id = (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));

-- Delete Keith's entry for the RM Winter Series Caulfield 2025 carnival
DELETE FROM ENTRY
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112')
  AND event_id IN (SELECT event_id FROM EVENT WHERE carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));

-- Delete the Super Runners team for the RM Winter Series Caulfield 2025 carnival
DELETE FROM TEAM
WHERE team_id = (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'));
COMMIT;