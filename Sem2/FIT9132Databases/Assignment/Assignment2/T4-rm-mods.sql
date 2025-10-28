--****PLEASE ENTER YOUR DETAILS BELOW****
--T4-rm-mods.sql

--Student ID: 27030768
--Student Name: Adrian Leong Tat Wei

/* Comments for your marker:




*/

--(a) Store number of completed events per competitor
SET TRANSACTION READ WRITE;

-- Add new column to COMPETITOR table
ALTER TABLE COMPETITOR
ADD (comp_completed_events_no NUMBER(3) DEFAULT 0);

COMMENT ON COLUMN COMPETITOR.comp_completed_events_no IS
    'Number of completed events for the competitor';

-- Populate the new column
UPDATE COMPETITOR c
SET comp_completed_events_no = (
    SELECT COUNT(e.entry_no)
    FROM ENTRY e
    WHERE e.comp_no = c.comp_no
      AND e.entry_finishtime IS NOT NULL
);

-- Show table structure change
DESC COMPETITOR;

-- Show data changes
SELECT comp_no, comp_fname, comp_lname, comp_completed_events_no FROM COMPETITOR;

COMMIT;

--(b) Add charity split support
SET TRANSACTION READ WRITE;

-- Drop before creating new table as good practice
DROP TABLE ENTRY_CHARITY_SUPPORT CASCADE CONSTRAINTS PURGE;

-- Create the new linking table
CREATE TABLE ENTRY_CHARITY_SUPPORT (
    event_id   NUMBER(6) NOT NULL,
    entry_no   NUMBER(5) NOT NULL,
    char_id    NUMBER(3) NOT NULL,
    percentage NUMBER(3) NOT NULL -- 0 to 100
    
);


-- Add foreign key constraints to the new table
ALTER TABLE ENTRY_CHARITY_SUPPORT ADD CONSTRAINT entry_charity_pk PRIMARY KEY (event_id, entry_no, char_id);

ALTER TABLE ENTRY_CHARITY_SUPPORT ADD CONSTRAINT entry_charity_chk CHECK (percentage BETWEEN 0 AND 100);

ALTER TABLE ENTRY_CHARITY_SUPPORT
ADD CONSTRAINT ecs_entry_fk FOREIGN KEY (event_id, entry_no)
    REFERENCES ENTRY (event_id, entry_no);

ALTER TABLE ENTRY_CHARITY_SUPPORT
ADD CONSTRAINT ecs_charity_fk FOREIGN KEY (char_id)
    REFERENCES CHARITY (char_id);

-- Migrate existing data from ENTRY.char_id to ENTRY_CHARITY_SUPPORT
INSERT INTO ENTRY_CHARITY_SUPPORT (event_id, entry_no, char_id, percentage)
SELECT event_id, entry_no, char_id, 100
FROM ENTRY
WHERE char_id IS NOT NULL;
COMMIT; -- Migration

-- Drop the old char_id column from ENTRY
ALTER TABLE ENTRY
DROP COLUMN char_id;
COMMIT; -- Structural change

-- Populate example data for Jackson Bull (for his 5K entry)
-- First, ensure previous Jackson's 5K entry charity is removed from ENTRY_CHARITY_SUPPORT
-- before inserting the new split percentages. This is important if his initial 5K entry
-- was already migrated with 100% to Beyond Blue.
DELETE FROM ENTRY_CHARITY_SUPPORT
WHERE event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
                 WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
                   AND et.eventtype_desc = '5 Km Run')
  AND entry_no = (SELECT en.entry_no FROM ENTRY en JOIN COMPETITOR c ON en.comp_no = c.comp_no
                 WHERE c.comp_phone = '0422412524'
                   AND en.event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
                                     WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
                                       AND et.eventtype_desc = '5 Km Run'));

-- Insert Jackson Bull's 70% to RSPCA
INSERT INTO ENTRY_CHARITY_SUPPORT (event_id, entry_no, char_id, percentage)
VALUES (
    (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
     WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
       AND et.eventtype_desc = '5 Km Run'),
    (SELECT en.entry_no FROM ENTRY en JOIN COMPETITOR c ON en.comp_no = c.comp_no
     WHERE c.comp_phone = '0422412524'
       AND en.event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
                           WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
                             AND et.eventtype_desc = '5 Km Run')),
    (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA'),
    70
);

-- Insert Jackson Bull's 30% to Beyond Blue
INSERT INTO ENTRY_CHARITY_SUPPORT (event_id, entry_no, char_id, percentage)
VALUES (
    (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
     WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
       AND et.eventtype_desc = '5 Km Run'),
    (SELECT en.entry_no FROM ENTRY en JOIN COMPETITOR c ON en.comp_no = c.comp_no
     WHERE c.comp_phone = '0422412524'
       AND en.event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
                           WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
                             AND et.eventtype_desc = '5 Km Run')),
    (SELECT char_id FROM CHARITY WHERE char_name = 'Beyond Blue'),
    30
);

COMMIT; -- Redistributed charity

-- Show table structure change for ENTRY
DESC ENTRY;

-- Show table structure change for new table
DESC ENTRY_CHARITY_SUPPORT;

-- Show data changes in ENTRY_CHARITY_SUPPORT for Jackson Bull's entry
SELECT ecs.event_id, ecs.entry_no, c.char_name, ecs.percentage
FROM ENTRY_CHARITY_SUPPORT ecs
JOIN CHARITY c ON ecs.char_id = c.char_id
JOIN ENTRY en ON ecs.event_id = en.event_id AND ecs.entry_no = en.entry_no
JOIN COMPETITOR comp ON en.comp_no = comp.comp_no
WHERE comp.comp_phone = '0422412524'
  AND en.event_id IN (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code
                      WHERE e.carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')
                        AND et.eventtype_desc = '5 Km Run');



