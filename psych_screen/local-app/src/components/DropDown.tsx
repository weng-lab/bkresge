import React, { useEffect, useState } from 'react';


type DropDownProps = {
    samples: string[];
    showDropDown: boolean;
    toggleDropDown: Function;
    sampleSelection: Function;
  };

  const DropDown: React.FC<DropDownProps> = ({
    samples,
    sampleSelection,
  }: DropDownProps): JSX.Element => {
    const [showDropDown, setShowDropDown] = useState<boolean>(false);
  
    /**
     * Handle passing the sample name
     * back to the parent component
     *
     * @param sample  The selected sample
     */
    const onClickHandler = (city: string): void => {
      sampleSelection(city);
    };
  
    useEffect(() => {
      setShowDropDown(showDropDown);
    }, [showDropDown]);
  
    return (
      <>
        <div className={showDropDown ? 'dropdown' : 'dropdown active'}>
          {samples.map(
            (sample: string, index: number): JSX.Element => {
              return (
                <p
                  key={index}
                  onClick={(): void => {
                    onClickHandler(sample);
                  }}
                >
                  {sample}
                </p>
              );
            }
          )}
        </div>
      </>
    );
  };
  
  export default DropDown;